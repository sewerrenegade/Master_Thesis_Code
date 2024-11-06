import math
import random
import sys

from datasets.mil_dataset_abstraction import BaseMILDataset
from distance_functions.functions.basic_distance_functions import EuclideanDistance, L1Distance
from datasets.SCEMILA import *

import numpy as np
import concurrent.futures
from tqdm import tqdm

AUGMENTATIONS_OF_INTEREST = ['all','none']
descriptors = []


def calculate_distance_matrix_of_MIL_dataset(dataset, nb_of_grouped_bags,distance_function,embedding_function, show_progress=True):
    if distance_function is None:
        distance_function = EuclideanDistance()
    bags_per_distance_matrix = nb_of_grouped_bags
    dataset = dataset
    assert isinstance(dataset, BaseMILDataset)
    
    #check result if exist
    
    #if not clculate
    total_nb_of_bags = len(dataset)
    subsets_for_dist_matrix_calculation = generate_subsets(total_nb_of_bags,bags_per_distance_matrix)
    bag_distance_matrix = {}
    bag_instance_order = {}
    print(subsets_for_dist_matrix_calculation)
    for subset_indices in subsets_for_dist_matrix_calculation:
        subsampled_bags = dataset[subset_indices][0]
        bag_instance_order.update({i:dataset.get_image_paths_from_index(i) for i in subset_indices})
        single_np_array = concatenate_arrays(subsampled_bags)
        if distance_function.name != "Cubical Complex Distance":
            single_np_array = single_np_array.reshape(single_np_array.shape[0], -1)
        embedded_points = embedding_function(single_np_array)
        split_embeddings = np.split(embedded_points, np.cumsum([len(bag) for bag in subsampled_bags])[:-1])
        
        for idx, instance_id in enumerate(subset_indices):
            if instance_id not in bag_distance_matrix:
                bag_embeddings = split_embeddings[idx]
                n_samples = len(bag_embeddings)
                distance_matrix = np.zeros((n_samples, n_samples))
                
                if show_progress:
                    progress_bar = tqdm(total=(n_samples * (n_samples - 1)) // 2, desc=f"Calculating distances for bag {instance_id}")

                with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:#ThreadPoolExecutor
                    futures = {}
                    for i in range(n_samples):
                        for j in range(i + 1, n_samples):
                            futures[(i, j)] = executor.submit(distance_function, bag_embeddings[i], bag_embeddings[j])
                    for (i, j), future in futures.items():
                        distance = future.result()
                        distance_matrix[i, j] = distance
                        distance_matrix[j, i] = distance
                        if show_progress:
                            progress_bar.update(1)

                    if show_progress:
                        progress_bar.close()
                
                bag_distance_matrix[instance_id] = distance_matrix
    assert len(bag_distance_matrix.keys()) == total_nb_of_bags
    assert len(bag_instance_order.keys()) == total_nb_of_bags     
    #save then load anyway
    return bag_distance_matrix,bag_instance_order

def concatenate_arrays(arrays):
    # Ensure the list is not empty
    if not arrays:
        raise ValueError("The input list is empty.")
    
    # Check if all arrays have the same shape for dimensions b, c, d
    reference_shape = arrays[0].shape[1:]  # Shape of dimensions b, c, d
    for array in arrays:
        if array.shape[1:] != reference_shape:
            raise ValueError("All arrays must have the same shape for dimensions b, c, d.")
    
    # Concatenate along the first dimension
    concatenated_array = np.concatenate(arrays, axis=0)
    
    return concatenated_array

def generate_subsets(n,nb_of_bags_per_calc):
    m = math.ceil(n / nb_of_bags_per_calc)
    initial_set = list(range(n))
    random.shuffle(initial_set)
    subsets = []
    for i in range(m - 1):
        subsets.append(initial_set[i*nb_of_bags_per_calc : (i+1)*nb_of_bags_per_calc])
    if n % nb_of_bags_per_calc == 0:
        last_subset = initial_set[(m-1)*nb_of_bags_per_calc : m*nb_of_bags_per_calc]
    else:
        recycled_elements_needed = (m*nb_of_bags_per_calc) - n
        recycled_elements = random.sample(range(nb_of_bags_per_calc), recycled_elements_needed)
        last_subset = initial_set[(m-1)*nb_of_bags_per_calc:] + [initial_set[i] for i in recycled_elements]
        # remaining_elements = [x for subset in subsets for x in subset]
        # resampled_elements = random.sample(remaining_elements, nb_of_bags_per_calc - unique_elements_needed)
        
        # last_subset = unique_elements + resampled_elements
    
    subsets.append(last_subset)
    
    return subsets
