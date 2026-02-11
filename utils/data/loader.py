import os
import json
import pickle
import numpy as np
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def calculate_local_mean(feature, time_window_size=10):
	"""
	Calculate local mean for a given feature around a current index.

	Args:
		feature (np.ndarray): The input feature array.
		time_window_size (int): The size of the time window to consider on each side of the current index.
	"""
	# local_min = np.min(feature[cur - time_window_size: cur + time_window_size])
	local_mean = np.array([np.mean(feature[max(cur - time_window_size, 0): cur + time_window_size]) for cur in range(len(feature))])
	return local_mean


def remove_specific_leading_chars(input_string):
    import re
    # Remove leading commas
    input_string = re.sub(r'^,+', '', input_string)
    # Remove numbers followed by a comma
    return re.sub(r'^\d+,+', '', input_string)

class BaseDataset:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.logger = logging.getLogger(__name__)   


    def load_raw_data(self):
        raise NotImplementedError


class MedQADataset(BaseDataset):
    def __init__(self, raw_data_path, processed_data_path):

        dataset_name = "MedLFQAv2"
        super().__init__(os.path.join(raw_data_path, dataset_name), os.path.join(processed_data_path, dataset_name))

        self.dataset_lookup = self.get_dataset_lookup()

        if os.path.exists(self.processed_data_path):
            with open(os.path.join(self.processed_data_path, 'dataset.pkl'), 'rb') as file:
                dataset = pickle.load(file)
            with open(os.path.join(self.processed_data_path, 'frequencies.pkl'), 'rb') as file:
                frequencies_arr = pickle.load(file)
            with open(os.path.join(self.processed_data_path, 'selfevals.pkl'), 'rb') as file:
                selfevals_arr = pickle.load(file)
            with open(os.path.join(self.processed_data_path, 'logprobs.pkl'), 'rb') as file:
                logprobs_arr = pickle.load(file)
            with open(os.path.join(self.processed_data_path, 'annotations.pkl'), 'rb') as file:
                annotations_arr = pickle.load(file)
            with open(os.path.join(self.processed_data_path, 'ordinal.pkl'), 'rb') as file:
                ordinal_arr = pickle.load(file)
            embeddings = np.load(os.path.join(self.processed_data_path, "embeddings.npy"))

            self.logger.info(f"Loaded processed data from {self.processed_data_path}")
        else:
            os.makedirs(self.processed_data_path)
            dataset, frequencies_arr, selfevals_arr, logprobs_arr, annotations_arr, ordinal_arr, embeddings = self.load_raw_data()

            with open(os.path.join(self.processed_data_path, 'dataset.pkl'), 'wb') as file:
                pickle.dump(dataset, file)
            with open(os.path.join(self.processed_data_path, 'frequencies.pkl'), 'wb') as file:
                pickle.dump(frequencies_arr, file)
            with open(os.path.join(self.processed_data_path, 'selfevals.pkl'), 'wb') as file:
                pickle.dump(selfevals_arr, file)
            with open(os.path.join(self.processed_data_path, 'logprobs.pkl'), 'wb') as file:
                pickle.dump(logprobs_arr, file)
            with open(os.path.join(self.processed_data_path, 'annotations.pkl'), 'wb') as file:
                pickle.dump(annotations_arr, file)
            with open(os.path.join(self.processed_data_path, 'ordinal.pkl'), 'wb') as file:
                pickle.dump(ordinal_arr, file)
            np.save(os.path.join(self.processed_data_path, "embeddings"), embeddings)

        self.dataset = dataset
        self.frequencies_arr = frequencies_arr
        self.selfevals_arr = selfevals_arr
        self.logprobs_arr = logprobs_arr
        self.annotations_arr = annotations_arr
        self.ordinal_arr = ordinal_arr
        self.embeddings = embeddings

        features = self.load_prompt_level_features()
        self.prompt_level_features = features
    
    def load_prompt_level_features(self):


        prompts_to_keep = [dat['prompt'] for dat in self.dataset]

        dataset_arr = [self.dataset_lookup[remove_specific_leading_chars(p).strip()] for p in prompts_to_keep]
        dataset_dummies = pd.get_dummies(dataset_arr)
        # dataset_names = [name[:-3] if name.endswith("_qa") else name for name in dataset_dummies.columns]

        # length of response
        response_len_arr = [len(dat['response']) for dat in self.dataset]
        response_len_arr = np.asarray(response_len_arr).reshape(-1,1)

        # length of prompt
        prompt_len_arr = [len(remove_specific_leading_chars(dat['prompt']).strip()) for dat in self.dataset]
        prompt_len_arr = np.asarray(prompt_len_arr).reshape(-1,1)

        # mean (exponentiated) logprob 
        logprobs_mean_arr = [np.mean(arr) for arr in self.logprobs_arr]
        logprobs_mean_arr = np.asarray(logprobs_mean_arr).reshape(-1,1)

        # std (exponentiated) logprob
        logprobs_std_arr = [np.std(arr) for arr in self.logprobs_arr]
        logprobs_std_arr = np.asarray(logprobs_std_arr).reshape(-1,1)

        z_arr = np.concatenate((response_len_arr, prompt_len_arr, logprobs_mean_arr, logprobs_std_arr), axis=1)
        z_dummies = dataset_dummies.to_numpy()

        z_arr_dummies = np.concatenate((z_arr, z_dummies), axis=1)
        return z_arr_dummies

    def load_raw_data(self):

        dataset_path = os.path.join(self.raw_data_path, "medlfqa_dataset.pkl")
        freq_path = os.path.join(self.raw_data_path, "medlfqa_frequencies.npz")
        logprob_path = os.path.join(self.raw_data_path, "medlfqa_logprobs.npz")
        selfeval_path = os.path.join(self.raw_data_path, "medlfqa_selfevals.npz")

        with open(dataset_path, 'rb') as fp:
            dataset = pickle.load(fp)


        ## HARD CODED FIX FOR REDUNDANT PROMPT...which has atomic_facts assigned to the wrong redundancy
        dataset[1132]['atomic_facts'] = dataset[1048]['atomic_facts']

        frequencies = np.load(freq_path)
        logprobs = np.load(logprob_path)
        selfevals = np.load(selfeval_path)

        drop_prompts = []
        for k in frequencies:
            if frequencies[k].ndim != 1:
                drop_prompts.append(k)
            elif np.allclose(selfevals[k], -1):
                drop_prompts.append(k)
            elif k not in logprobs:
                drop_prompts.append(k)
            elif remove_specific_leading_chars(k).strip() not in self.dataset_lookup:
                drop_prompts.append(k)

        # drop and match ordering of dataset
        dataset = [dat for dat in dataset if dat['prompt'] not in drop_prompts]
        # full_dataset = dataset
        prompts_to_keep = [dat['prompt'] for dat in dataset]
        # names_to_keep = [p.split('about')[-1].strip()[:-1] for p in prompts_to_keep]

        frequencies_arr = [frequencies[p] for p in prompts_to_keep]
        selfevals_arr = [selfevals[p] for p in prompts_to_keep]
        logprobs_arr = [logprobs[p] for p in prompts_to_keep]
        annotations_arr = [np.asarray([af["is_supported"] for af in dat["atomic_facts"]]) for dat in dataset]
        ordinal_arr = [np.arange(len(f)) for f in frequencies_arr]

        responses = [dat["response"] for dat in dataset]    
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(responses)

        return dataset, frequencies_arr, selfevals_arr, logprobs_arr, annotations_arr, ordinal_arr, embeddings
    
    def get_dataset_lookup(self):
        orig_datasets = {}
        suffix = '.jsonl'
        dataset_dir = os.path.join(self.raw_data_path, "orig_data")
        for path in os.listdir(dataset_dir):
            dataset_name = path[:-len(suffix)]
            with open(os.path.join(dataset_dir, path), 'r') as fp:
                orig_datasets[dataset_name] = [json.loads(line) for line in fp.readlines()]

        dataset_lookup = {}
        for name, data in orig_datasets.items():
            for dat in data:
                dataset_lookup[dat['Question']] = name

        return dataset_lookup

class WikiDataset(BaseDataset):

    def __init__(self, raw_data_path, processed_data_path):

        dataset_name = "Wiki"
        super().__init__(os.path.join(raw_data_path, dataset_name), os.path.join(processed_data_path, dataset_name))

        if os.path.exists(self.processed_data_path):
            with open(os.path.join(self.processed_data_path, 'dataset.pkl'), 'rb') as file:
                dataset = pickle.load(file)
            with open(os.path.join(self.processed_data_path, 'frequencies.pkl'), 'rb') as file:
                frequencies_arr = pickle.load(file)
            with open(os.path.join(self.processed_data_path, 'selfevals.pkl'), 'rb') as file:
                selfevals_arr = pickle.load(file)
            with open(os.path.join(self.processed_data_path, 'annotations.pkl'), 'rb') as file:
                annotations_arr = pickle.load(file)
            embeddings = np.load(os.path.join(self.processed_data_path, "embeddings.npy"))

            meta_data = pd.read_csv(os.path.join(self.processed_data_path, "metadata.csv"))

            self.logger.info(f"Loaded processed data from {self.processed_data_path}")
        else:
            os.makedirs(self.processed_data_path)
            dataset, frequencies_arr, selfevals_arr, annotations_arr, meta_data, embeddings = self.load_raw_data()

            with open(os.path.join(self.processed_data_path, 'dataset.pkl'), 'wb') as file:
                pickle.dump(dataset, file)
            with open(os.path.join(self.processed_data_path, 'frequencies.pkl'), 'wb') as file:
                pickle.dump(frequencies_arr, file)
            with open(os.path.join(self.processed_data_path, 'selfevals.pkl'), 'wb') as file:
                pickle.dump(selfevals_arr, file)
            with open(os.path.join(self.processed_data_path, 'annotations.pkl'), 'wb') as file:
                pickle.dump(annotations_arr, file)
            np.save(os.path.join(self.processed_data_path, "embeddings"), embeddings)

            meta_data.to_csv(os.path.join(self.processed_data_path, "metadata.csv"), index=False)



        self.dataset = dataset
        self.frequencies_arr = frequencies_arr
        self.selfevals_arr = selfevals_arr
        self.annotations_arr = annotations_arr
        self.meta_data = meta_data
        self.embeddings = embeddings

        features = self.get_prompt_level_features()
        self.prompt_level_features = features


    def load_raw_data(self):
        with open(os.path.join(self.raw_data_path, "factscore_final_dataset.pkl"), 'rb') as fp:
            dataset = pickle.load(fp)
        
        self.dataset = dataset

        frequencies = np.load(os.path.join(self.raw_data_path, "factscore_final_frequencies_2.npz"))
        selfevals = np.load(os.path.join(self.raw_data_path, "factscore_final_self_evals.npz"))
        metadata = pd.read_csv(os.path.join(self.raw_data_path, "factscore_final.csv"), index_col=0).reset_index(drop=True).drop_duplicates()

        drop_prompts = []
        for k in frequencies:
            if frequencies[k].ndim != 1:
                drop_prompts.append(k)
            elif np.allclose(selfevals[k], -1):
                drop_prompts.append(k)

        # drop and match ordering of dataset
        dataset = [dat for dat in dataset if dat['prompt'] not in drop_prompts]

        prompts_to_keep = [dat['prompt'] for dat in dataset]
        names_to_keep = [p.split('about')[-1].strip()[:-1] for p in prompts_to_keep]

        metadata = metadata.set_index("Name")
        metadata = metadata.loc[names_to_keep].reset_index()

        metadata.loc[663, "count_bins"] = "Very Rare"
        metadata.loc[663, "Views"] = 0

        bins = [0, 100, 1000, 10000, 100000, np.inf]
        names = ['Very Rare', 'Rare', 'Medium', 'Frequent', 'Very Frequent']

        metadata['count_bins'] = pd.cut(metadata['Views'], bins, labels=names)

        frequencies_arr = [frequencies[p] for p in prompts_to_keep]
        selfevals_arr = [selfevals[p] for p in prompts_to_keep]
        annotations_arr = [np.asarray([af["is_supported"] for af in dat["atomic_facts"]]) for dat in dataset]

        responses = [dat["response"] for dat in dataset]
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(responses)


        return dataset, frequencies_arr, selfevals_arr, annotations_arr, metadata, embeddings

    def get_prompt_level_features(self):

        z_ones = np.ones((len(self.frequencies_arr), 1))
        views = self.meta_data["Views"].to_numpy()
        views += 1
        z_views = views.clip(0, np.quantile(views, 0.95)).reshape(-1,1)
        z_views = z_views / np.mean(z_views)

        z_arr = np.concatenate((z_ones, z_views, z_views**2, z_views**3), axis=1)

        return z_arr





def load_dataset(name, raw_data_path, processed_data_path, **kwargs):
    if name == "MedQA":
        dataset = MedQADataset(raw_data_path, processed_data_path)
    elif name == "Wiki":
        dataset = WikiDataset(raw_data_path, processed_data_path)
    else:
        raise ValueError(f"No such dataset: {name}")

    return dataset.frequencies_arr, dataset.annotations_arr, dataset.prompt_level_features, dataset.embeddings

def split_dataset(dataset, rng, train_frac):
    # Get the number of samples from the first element of the tuple.
    # This assumes all elements have the same number of samples.
    total_samples = len(dataset[0])

    # Shuffle indices to create a random split.
    ind = np.arange(total_samples)
    rng.shuffle(ind)

    # Calculate split point and create train and calibration indices.
    train_num = int(train_frac * total_samples)
    train_ind = ind[0:train_num]
    calib_ind = ind[train_num:]

    # Use list comprehensions to split each element in the input tuple.
    train_split = []
    calib_split = []

    for item in dataset:
        # Check if the item is a numpy array to use efficient slicing.
        if isinstance(item, np.ndarray):
            train_split.append(item[train_ind])
            calib_split.append(item[calib_ind])
        # Otherwise, assume it is a list or similar and use list comprehension.
        else:
            train_split.append([item[i] for i in train_ind])
            calib_split.append([item[i] for i in calib_ind])

    return train_split, calib_split