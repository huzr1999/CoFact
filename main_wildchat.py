from utils.logger import init_logger, get_logger
from utils.data.loader import MedQADataset, WikiDataset, load_dataset, split_dataset
from utils.data.dataset import TrainSet, TestSet, normalize_features, reduce_dimensions
import yaml
import numpy as np
from CP.conformal_prediction import run_split_conformal, score_func, get_retained_claims
from tqdm import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
import torch
import random
from conditionalconformal import CondConf
import umap
from online.offline_training import offline_train
import glob
from online.estimator.accous import Accous
from online.model import Linear
from utils.tools import Timer
import settings
import pandas as pd
import math

def create_batches_comprehension(input_list, batch_size):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

def get_testset(data_test, per_step_num):

    test_batched_idx = create_batches_comprehension(list(range(len(data_test[0]))), batch_size=per_step_num)
    # test_set = list(zip(data_test[0], data_test[1], data_test[2], data_test[3], range(len(data_test[0]))))
    # test_set_batched = create_batches_comprehension(test_set, batch_size=per_step_num)
    for batch in test_batched_idx:
        yield (
            [data_test[0][i] for i in batch],
            [data_test[1][i] for i in batch],
            data_test[2][batch],
            data_test[3][batch],
            batch
        )

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def split_dataset_wildchat(*args, train_frac):
    """
    Splits an arbitrary number of arrays into a calibration and a test set.

    Args:
        train_frac (float): The fraction of the dataset to use for the 
                            calibration (training) set.
        *args: A variable number of NumPy arrays or lists of the same length
               to be split.

    Returns:
        tuple: A tuple containing two lists. The first list holds the
               calibration sets, and the second list holds the test sets.
               The order of the returned arrays will match the order of the
               input arrays.
    """
    if not args:
        print("Warning: No arrays were provided to the function.")
        return [], []

    # Get the length from the first array, assuming all have the same length
    num_total = len(args[0])
    num_calib = int(train_frac * num_total)
    
    # Initialize lists for the results
    calib_sets = []
    test_sets = []
    
    # Iterate through all provided arrays and split them
    for arr in args:
        calib_sets.append(arr[:num_calib])
        test_sets.append(arr[num_calib:])
        
    return calib_sets, test_sets


def run_coverage_trial_unconditional(score_arr, annotation_arr, prompt_level_features, responses_rep, k, quantile, per_step_num, train_frac):


    data_calib, data_test = split_dataset_wildchat(score_arr, annotation_arr, prompt_level_features, responses_rep, train_frac=train_frac)


    data_calib[2], data_test[2] = normalize_features(data_calib[2], data_test[2])

    # data_calib[3], data_test[3] = reduce_dimensions(data_calib[3], data_test[3], n_components=reduced_dim)
    # data_calib[3] = normalize_features(data_calib[3])
    # data_test[3] = normalize_features(data_test[3])

    _, threshold = run_split_conformal(data_calib[0], data_calib[1], method=k, quantile=quantile)

    test_set = get_testset(data_test, per_step_num)

    coverages_one_trial = []
    claims_one_trial = []
    for S_t in test_set:

        scores_test = score_func(S_t[0], S_t[1], method=k)
        valid_inds = scores_test <= threshold

        claim_perc = []
        for scores in S_t[0]:
            claim_perc.append(np.mean(scores > threshold))

        valid_inds = np.asarray(valid_inds).flatten()
        claim_perc = np.asarray(claim_perc).flatten()

        coverages_one_trial.append(np.mean(valid_inds))
        claims_one_trial.append(np.mean(claim_perc))
    
    return coverages_one_trial, claims_one_trial

def run_coverage_trial_conditional(score_arr, annotation_arr, prompt_level_features, responses_reps, quantile, k, per_step_num, estimator_base, train_frac):

    data_calib, data_test = split_dataset_wildchat(score_arr, annotation_arr, prompt_level_features, responses_reps, train_frac=train_frac)


    data_calib[2], data_test[2] = normalize_features(data_calib[2], data_test[2])

    data_test = [data_test[0], data_test[1], data_test[2], data_test[3]]


    scores_calib = score_func(data_calib[0], data_calib[1], method=k)
    # scores_test = score_func(data_test[0], data_test[2], method=k)

    condconf = CondConf(lambda x,y: y, lambda x: x)

    if estimator_base == "prompt_level_features":
        condconf.setup_problem(data_calib[2], scores_calib)
    elif estimator_base == "responses_rep":
        condconf.setup_problem(data_calib[3], scores_calib)
    else:
        raise ValueError(f"No such estimator base: {estimator_base}")

    test_set = get_testset(data_test, per_step_num)


    coverages_one_trial = []
    claims_one_trial = []

    for S_t in test_set:

        scores_test = score_func(S_t[0], S_t[1], method=k)

        valid_inds = []
        claim_perc = []

        for i, (score, y, x, rep, inds) in enumerate(zip(*S_t)):
            # x = x[1:]  # remove the intercept term  
            try:
                if estimator_base == "prompt_level_features":
                    threshold = condconf.predict(quantile, x.reshape(1,-1), lambda c, x: c, randomize=True)
                elif estimator_base == "responses_rep":
                    threshold = condconf.predict(quantile, rep.reshape(1,-1), lambda c, x: c, randomize=True)
                else:
                    raise ValueError(f"No such estimator base: {estimator_base}")
            except:
                threshold = [np.inf]

            if not isinstance(threshold >= scores_test[i], np.ndarray):
                valid_inds.append(np.array([threshold >= scores_test[i]]))
            else:
                valid_inds.append(threshold >= scores_test[i])
            claim_perc.append(get_retained_claims([score], [threshold])[0])

        try:
            valid_inds = np.asarray(valid_inds).flatten()
            claim_perc = np.asarray(claim_perc).flatten()
        except:
            import IPython; IPython.embed()

        coverages_one_trial.append(np.mean(valid_inds))
        claims_one_trial.append(np.mean(claim_perc))
    
    return coverages_one_trial, claims_one_trial

def run_coverage_trial_online(frequencies_arr, annotations_arr, prompt_level_features, response_rep, dataset_labels, seed, quantile, k, per_step_num, estimator_base, train_frac, init_model_cfg, online_cfg):


    logger = get_logger(__name__)


    train_data, test_data = split_dataset_wildchat(frequencies_arr, annotations_arr, prompt_level_features, response_rep, dataset_labels, train_frac=train_frac)


    train_scores, train_Y, train_prompt_level_features, train_X, train_ds_labels = train_data
    test_scores, test_Y, test_prompt_level_features, test_X, test_ds_labels = test_data


    train_prompt_level_features, test_prompt_level_features = normalize_features(train_prompt_level_features, test_prompt_level_features)

    test_data = (test_scores, test_Y, test_prompt_level_features, test_X, test_ds_labels)


    features_used_for_training = train_prompt_level_features if estimator_base == "prompt_level_features" else train_X
    if init_model_cfg["load_pretrained"]:
        subdirectories = glob.glob(os.path.join(init_model_cfg["saved_model_path"], "*/"))

        latest_subdir = max(subdirectories, key=os.path.getmtime)

        model_files = glob.glob(os.path.join(latest_subdir, '*.pth'))

        if len(model_files) > 0:
            # Get the full path to the model file
            model_path = model_files[0]

            init_model = Linear(
                # input_dim=train_X.shape[1],
                input_dim=features_used_for_training.shape[1],
                output_dim=init_model_cfg['offline_training']['num_cls'],
                R=init_model_cfg['offline_training']['R']
            )
            init_model.load_state_dict(torch.load(model_path, map_location=init_model_cfg['device']))
            estimator_model = init_model
            logger.info(f"Loaded latest pretrained model from {model_path}")
        else:
            # logger.warning("No .pth file found in the latest subdirectory.")
            raise FileNotFoundError("No .pth file found in the latest subdirectory.")

    else:
        # init_model = offline_train(train_X, train_ds_labels, device=init_model_cfg['device'], saved_model_path=init_model_cfg["saved_model_path"], **init_model_cfg["offline_training"])
        init_model = offline_train(features_used_for_training, train_ds_labels, device=init_model_cfg['device'], saved_model_path=init_model_cfg["saved_model_path"], **init_model_cfg["offline_training"])
        estimator_model = init_model


    if estimator_base == "responses_rep":
        dim = train_X.shape[1] 
    elif estimator_base == "prompt_level_features":
        dim = train_prompt_level_features.shape[1] 
    else:
        raise ValueError(f"No such estimator base: {estimator_base}")

    online_cfg['T'] = math.ceil(len(test_scores) / per_step_num)
    algorithm = Accous(cfgs=online_cfg, dim=dim, seed=seed, model=estimator_model)

    test_set = get_testset(test_data, per_step_num)


    tr_rep = torch.from_numpy(train_X).float()
    
    tr_conformal_scores = score_func(train_scores, train_Y, method=k)

    coverages_one_trial = []
    claims_one_trial = []
    

    loss_vectors = []
    acc_vectors = []

    for t, S_t in enumerate(tqdm(test_set, total=len(test_scores) // per_step_num + 1)):

        te_scores, te_label, te_prompt_level_features, te_rep, te_sampled_inds = S_t
        # print(te_scores)
        te_rep = torch.tensor(te_rep).float()

        te_conformal_scores = score_func(te_scores, te_label, method=k)

        if estimator_base == "prompt_level_features":
            offline_rep = torch.from_numpy(train_prompt_level_features).float()
            online_rep = torch.from_numpy(te_prompt_level_features)
        elif estimator_base == "responses_rep":
            offline_rep = torch.Tensor.cpu(tr_rep).float()
            online_rep = torch.Tensor.cpu(te_rep)
        else:
            raise ValueError(f"No such estimator base: {estimator_base}")

        weights, loss_vector, acc_vector = algorithm.predict_and_update(offline_rep, online_rep, torch.concat([offline_rep, online_rep], dim=0))

        weights = weights.cpu().numpy()


        loss_vectors.append(loss_vector.cpu().numpy())
        acc_vectors.append(np.array(acc_vector))

        train_size, test_t_size = len(train_scores), len(te_scores)

        validity_at_t = []
        claims_perc_at_t = []
        for i in range(train_size, train_size + test_t_size):

            local_weights = np.concatenate([weights[:train_size], weights[i:i+1]])


            aug_conformal_scores = np.concatenate([tr_conformal_scores, np.array([np.inf])])
            sorted_indices = np.argsort(aug_conformal_scores)
            sorted_scores = aug_conformal_scores[sorted_indices]
            sorted_weights = local_weights[sorted_indices]

            assert len(sorted_indices) == len(local_weights)

            # Cumulative sum of weights
            cumulative_weights = np.cumsum(sorted_weights)
            
            # Find the index where the cumulative weight first exceeds alpha
            weighted_quantile_index = np.where(cumulative_weights >= quantile * np.sum(sorted_weights))[0][0]
                
                # The threshold is the score at this index
            threshold = sorted_scores[weighted_quantile_index]


            validity = threshold >= te_conformal_scores[i - train_size]

            # print(te_conformal_scores[i - train_size])
            claims_perc = get_retained_claims([te_scores[i - train_size]], [threshold])[0]

            validity_at_t.append(validity)
            claims_perc_at_t.append(claims_perc)


        coverages_one_trial.append(np.array(validity_at_t).mean())
        claims_one_trial.append(np.array(claims_perc_at_t).mean())

    return coverages_one_trial, claims_one_trial




def run_coverage_trials(scores_arr, annotations_arr, prompt_level_features, response_reps, dataset_labels, cfg):


    seed = cfg["random_seed"]
    num_trials = cfg["Experiment"]["kwargs"]["num_trials"]
    train_frac = cfg["Experiment"]["kwargs"]["train_frac"]
    method_type = cfg["Experiment"]["method"]
    quantile = cfg["Experiment"]["kwargs"]["quantile"]
    k = cfg["Experiment"]["kwargs"]["k"]
    per_step_num = cfg["Experiment"]["kwargs"]["per_step_num"]
    estimator_base = cfg["Experiment"]["estimator_base"]

    logger = logging.getLogger(__name__)


    rng = np.random.default_rng(seed=seed+1)

    all_covs = []
    all_claims = []

    for trials in range(num_trials):

        logger.info(f"Start one trial on {method_type}, the {trials+1}-th trial!")
        scores_arr_jitter = [score + rng.uniform(low=0, high=1e-3, size=score.shape) for score in scores_arr]

        if method_type == "CP-unconditional":
            coverages_one_trial, claims_one_trial = run_coverage_trial_unconditional(scores_arr_jitter, annotations_arr, prompt_level_features, response_reps, k, quantile, per_step_num, train_frac=train_frac)
        elif method_type == "CP-conditional":
            coverages_one_trial, claims_one_trial = run_coverage_trial_conditional(scores_arr_jitter, annotations_arr, prompt_level_features, response_reps,  quantile, k, per_step_num, estimator_base=estimator_base, train_frac=train_frac)
        elif method_type == "Online":
            coverages_one_trial, claims_one_trial = run_coverage_trial_online(scores_arr_jitter, annotations_arr, prompt_level_features, response_reps, dataset_labels, seed, quantile, k, per_step_num, estimator_base=estimator_base, train_frac=train_frac, init_model_cfg=cfg["InitModel"], online_cfg=cfg["OnlineLearning"])
        else:
            raise ValueError(f"No such method: {method_type}")


        all_covs.append(np.array(coverages_one_trial))
        all_claims.append(np.array(claims_one_trial))
        logger.info(f"One trial on {method_type} has done!")
        logger.info(f"The coverage of this trial is {all_covs[trials].mean():.3f}, the claim percentage of this trial is {all_claims[trials].mean():.3f}")
    
    all_covs = np.vstack(all_covs)
    all_claims = np.vstack(all_claims)

    return all_covs, all_claims




def plot_coverage_and_claims(all_covs, all_claims, window_size, saved_path):

    n_trials = all_covs.shape[0]
    num_time_steps = all_covs.shape[1]

    local_means = np.zeros((n_trials, num_time_steps))

    for trial in (range(n_trials)):
        for i in range(num_time_steps):
            start_idx = max(0, i - window_size)
            end_idx = min(num_time_steps - 1, i + window_size)

            # Calculate the mean of the slice
            local_means[trial, i] = np.mean(all_covs[trial, start_idx:end_idx + 1])

    mean = local_means.mean(axis=0)
    std_error = local_means.std(axis=0) / np.sqrt(n_trials)

    lower_bound = np.array(mean) - np.array(std_error)
    upper_bound = np.array(mean) + np.array(std_error)

    x = np.arange(num_time_steps)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, mean, color='orange', label='Local Mean')
    ax.fill_between(x, lower_bound, upper_bound, color='blue', alpha=0.2, label='Standard Error')


    ax.set_title('Local Mean of Each Element', fontsize=16)
    ax.set_xlabel('Element Index', fontsize=12)
    ax.set_ylabel('Local Mean', fontsize=12)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    fig.savefig(os.path.join(saved_path, "coverage_and_claims.pdf"))


if __name__ == "__main__":


    results_saved_path = settings.RESULTS_PATH
    cfg = settings.cfg
    
    setup_seed(cfg["random_seed"])
    

    init_logger(os.path.join(results_saved_path, "debug.log"), cfg["Logging"]["logger_file_level"])
    logger = get_logger(__name__, logging.DEBUG)

    logger.info(f"We are using config file loaded from {settings.config_path}")


    # frequencies_arr, annotations_arr, prompt_level_features = load_dataset(**cfg["Dataset"])
    frequencies_arr, annotations_arr, prompt_level_features, response_rep = load_dataset(**cfg['Dataset'])

    if cfg["Dataset"]["name"] == "MedQA":
        dataset_labels = prompt_level_features[:, -5:].argmax(axis=1)
    elif cfg["Dataset"]["name"] == "Wiki":
        # logger.info(prompt_level_features[:, -1])
        data_series = pd.Series(prompt_level_features[:, -1])
        # Discretize into 5 labels with equal-sized bins
        labels_series = pd.qcut(data_series, q=cfg["InitModel"]["offline_training"]["num_cls"], labels=False)
        # Convert back to a NumPy array
        dataset_labels = labels_series.to_numpy()
    elif cfg["Dataset"]["name"] == "WildChat":
        # logger.info(prompt_level_features[:, -1])
        data_series = pd.Series(prompt_level_features[:, -1])
        # Discretize into 5 labels with equal-sized bins
        labels_series = pd.qcut(data_series, q=cfg["InitModel"]["offline_training"]["num_cls"], labels=False)
        # Convert back to a NumPy array
        dataset_labels = labels_series.to_numpy()
    else:
        raise ValueError(f"No such dataset: {cfg['Dataset']['name']}")

    all_covs, all_claims = run_coverage_trials(frequencies_arr, annotations_arr, prompt_level_features, response_rep, dataset_labels, cfg)




    logger.info("The marginal coverage is %.3f ± %.3f", all_covs.mean(axis=1).mean(), all_covs.mean(axis=1).std())
    logger.info("The marginal claims are %.3f ± %.3f", all_claims.mean(axis=1).mean(), all_claims.mean(axis=1).std())

    plot_coverage_and_claims(all_covs, all_claims, **cfg["Plot"]["kwargs"], saved_path=results_saved_path)

    with open(os.path.join(results_saved_path, "results.json"), "w") as f:
       json.dump({"all_covs": all_covs.tolist(), "all_claims": all_claims.tolist()}, f)
       logger.info("Results saved to %s", os.path.join(results_saved_path, "results.json"))
    
    with open(os.path.join(results_saved_path, "config_used.yaml"), "w") as f:
       yaml.dump(cfg, f)
       logger.info("Config used saved to %s", os.path.join(results_saved_path, "config_used.yaml"))

    