from utils.logger import get_logger
import numpy as np


def sanity_check(dataset, frequencies, metadata):

    logger = get_logger(__name__)
    logger.info("================ Starting sanity check of the dataset, frequencies, and metadata.================")

    dataset_prompts = [data['prompt'] for data in dataset]
    metadata_prompts = metadata['prompt'].tolist()

    num_not_matches = 0
    for prompt1, prompt2 in zip(dataset_prompts, metadata_prompts):
        if prompt1 != prompt2:
            # print("Mismatch found:")
            # print("Dataset Prompt:", prompt1)
            # print("Metadata Prompt:", prompt2)
            num_not_matches += 1
    
    if num_not_matches == 0:
        logger.info("All prompts match between dataset and metadata.")
    else:
        logger.warning(f"Total not matching prompts between dataset and metadata: {num_not_matches} out of {len(dataset_prompts)}")

    if_prompt_unique = len(set(dataset_prompts)) == len(dataset_prompts)
    if if_prompt_unique:
        logger.info("All prompts in the dataset are unique.")
    else:
        logger.warning(f"There are {len(dataset_prompts) - len(set(dataset_prompts))} duplicate prompts in the dataset.")

    num_invalid_frequencies = 0
    for freq in frequencies:
        if not isinstance(freq, np.ndarray):
            num_invalid_frequencies += 1
    if num_invalid_frequencies == 0:
        logger.info("All frequencies are valid numpy arrays.")
    else:
        logger.warning(f"Total invalid frequencies (not numpy arrays): {num_invalid_frequencies} out of {len(frequencies)}")
    
    dataset_atomic_facts_nums = [len(data['atomic_facts']) for data in dataset]
    atomic_facts_nums = [len(freq) if isinstance(freq, np.ndarray) else 0 for freq in frequencies]

    num_not_matches = 0
    for num1, num2 in zip(dataset_atomic_facts_nums, atomic_facts_nums):
        if num1 != num2:
            # print("Mismatch in atomic facts number found:")
            # print("Dataset Atomic Facts Number:", num1)
            # print("Frequencies Atomic Facts Number:", num2)
            num_not_matches += 1
    
    if num_not_matches == 0:
        logger.info("All atomic facts numbers match between dataset and frequencies.")
    else:
        logger.warning(f"Total not matching atomic facts numbers between dataset and frequencies: {num_not_matches} out of {len(dataset_atomic_facts_nums)}")


    # annotations = [np.array([atom['is_supported'] for atom in dat['atomic_facts']]) for dat in dataset]

    none_annotation_count = 0
    for dat in dataset:

        none_annotation = False
        for i in range(len(dat['atomic_facts'])):
            if dat['atomic_facts'][i]['is_supported'] is None:
                none_annotation = True
                break
        if none_annotation:
                none_annotation_count += 1
    if none_annotation_count == 0:
        logger.info("No prompts with None annotations in the dataset.")
    else:
        logger.warning(f"Total prompts with None annotations: {none_annotation_count}")
    logger.info("================ Finished sanity check of the dataset, frequencies, and metadata.================")

def fix_data(dataset, frequencies, metadata):

    logger = get_logger(__name__)
    logger.info("================ Starting to fix the dataset, frequencies, and metadata.================")

    invalid_frequencies_indices = [i for i, freq in enumerate(frequencies) if not isinstance(freq, np.ndarray)]

    if invalid_frequencies_indices:
        logger.warning(f"Found {len(invalid_frequencies_indices)} invalid frequencies at indices: {invalid_frequencies_indices}. These entries will be removed.")
        dataset = [data for i, data in enumerate(dataset) if i not in invalid_frequencies_indices]
        frequencies = [freq for i, freq in enumerate(frequencies) if i not in invalid_frequencies_indices]
    
    invalid_none_annotation_indices = []
    for i, dat in enumerate(dataset):
        none_annotation = False
        for j in range(len(dat['atomic_facts'])):
            if dat['atomic_facts'][j]['is_supported'] is None:
                none_annotation = True
                break
        if none_annotation:
            invalid_none_annotation_indices.append(i)
    
    if invalid_none_annotation_indices:
        logger.warning(f"Found {len(invalid_none_annotation_indices)} entries with None annotations at indices: {invalid_none_annotation_indices}. These entries will be removed.")
        dataset = [data for i, data in enumerate(dataset) if i not in invalid_none_annotation_indices]
        frequencies = [freq for i, freq in enumerate(frequencies) if i not in invalid_none_annotation_indices]
        # metadata = metadata.drop(index=invalid_none_annotation_indices).reset_index(drop=True)

    dataset_prompts = [data['prompt'] for data in dataset]
    metadata_prompts = metadata['prompt'].tolist()

    fixed_dataset = []
    fixed_frequencies = []
    fixed_metadata_indices = []

    # dataset_prompts_set = set(dataset_prompts)

    # for i, prompt in enumerate(dataset_prompts_set):
    # 	if prompt in metadata_prompts:
    # 		fixed_dataset.append(dataset[i])
    # 		fixed_frequencies.append(frequencies[i])
    # 		fixed_metadata_indices.append(metadata_prompts.index(prompt))
    # 	else:
    # 		logger.warning(f"Prompt from dataset not found in metadata: {prompt[:50]}... Skipping this entry.")

    # fixed_metadata = metadata.iloc[fixed_metadata_indices].reset_index(drop=True)

    # 1. Map prompts to their first seen index to handle duplicates and preserve alignment
    unique_prompt_map = {}
    for idx, prompt in enumerate(dataset_prompts):
        if prompt not in unique_prompt_map:
            unique_prompt_map[prompt] = idx

# 2. Convert metadata_prompts to a dict/set for O(1) lookups
    meta_lookup = {p: i for i, p in enumerate(metadata_prompts)}

    for prompt, orig_idx in unique_prompt_map.items():
        if prompt in meta_lookup:
            fixed_dataset.append(dataset[orig_idx])
            fixed_frequencies.append(frequencies[orig_idx])
            # Use the pre-calculated lookup instead of .index()
            fixed_metadata_indices.append(meta_lookup[prompt])
        else:
            pass
            # logger.warning(f"Prompt not found in metadata...")

    # 3. Final alignment
    fixed_metadata = metadata.iloc[fixed_metadata_indices].reset_index(drop=True)



    logger.info(f"Fixed dataset size: {len(fixed_dataset)}")
    logger.info(f"Fixed frequencies size: {len(fixed_frequencies)}")
    logger.info(f"Fixed metadata size: {len(fixed_metadata)}")

    logger.info("================ Finished fixing the dataset, frequencies, and metadata.================")

    return fixed_dataset, fixed_frequencies, fixed_metadata
    
    