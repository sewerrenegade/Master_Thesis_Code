import numpy as np


def calculate_linear_affinity(distance_matrix):
    nb_classes = distance_matrix.shape[0]
    nb_off_diagonal_to_diag_ratio = nb_classes - 1
    assert distance_matrix.shape[0] == distance_matrix.shape[1]
    tst_mat = - np.eye(nb_classes)  + (np.ones((nb_classes,nb_classes)) - np.eye(nb_classes))/nb_off_diagonal_to_diag_ratio
    #soft_max = calculate_softmax(distance_matrix)
    row_means = np.mean(distance_matrix, axis=1)
    class_distance =1*  distance_matrix / row_means[:, np.newaxis]
    return np.sum(tst_mat * class_distance) + nb_classes

def calculate_crossentropy(distance_matrix):
    soft_max = calculate_softmax(distance_matrix)
    cross_entropy = -np.log(np.prod(np.diag(soft_max)))
    return cross_entropy

def calculate_variational_ratio(distance_matrix):
    soft_max = calculate_softmax(distance_matrix)
    variational_ratio = np.sum(1-np.diag(soft_max))/distance_matrix.shape[0]
    return variational_ratio

def get_score_of_distances(distance_mat,data_used):
    distance_mat = standardize_array_mean(distance_mat)
    linear_affinity = calculate_linear_affinity(distance_mat)
    cross_entropy = calculate_crossentropy(distance_mat)
    var_ratio = calculate_variational_ratio(distance_mat)
    k_neighbors_average_class_dissimilarity,k_neighbors_name = calculate_k_neighbors_class_dissimilarity(data_used,nb_neighbours=10)#needs to incorporate distance function TODO
   
    return {k_neighbors_name:k_neighbors_average_class_dissimilarity,"linear_affinity":linear_affinity,"cross_entropy":cross_entropy,"var_ratio":var_ratio}

def calculate_k_neighbors_class_dissimilarity(data_used,nb_per_class_sample_points=5,nb_neighbours = 5):
    data,lables = flatten_dict_and_create_labels(data_used)
    scores = []
    k_neighbors_name =f"k_{nb_neighbours}_neighbourhood_dissimilarity"
    for key,value in data_used.items():
        for target_point in value[:nb_per_class_sample_points]:
            closest_indicies = calculate_the_index_of_k_closest_points(data,target_point,nb_neighbours)
            closest_points_labels = [lables[i] for i in closest_indicies]
            scores.append(closest_points_labels.count(key)/len(closest_points_labels))
    return 1 - np.mean(np.array(scores)),k_neighbors_name

#retruns the indicies of top k closest neighbour points
def calculate_the_index_of_k_closest_points(data,target_point,nb_neighbours = 5):
    #other_points = np.delete(data, target_index, axis=0)
    distances = np.linalg.norm(data - target_point, axis=1)
    closest_indices = np.argsort(distances)[:nb_neighbours+1]
    closest_indices = list(closest_indices)
    if distances[closest_indices[0]] == 0:
        closest_indices.remove(closest_indices[0])
    else:
        print("very weird behaviour")
        closest_indices = closest_indices[:-1]
    return closest_indices


def flatten_dict_and_create_labels(data):
    labels = []
    flat_data = []
    for key,val in data.items():
        n = len(val)
        labels.extend([key]*n)
        flat_data.extend(val)
    return np.array(flat_data),labels

#makes it so the distribution of ALL distances to be 0 mean and have a variance of 1
def center_at_zero_and_normalise(array):
    avg = np.mean(array)
    std = np.std(array)
    return (array-avg)/std

#if true makes that average distance between a class and all others (including itself) equal to 1
#if false makes the average distance between all classes equal to 1
def standardize_array_mean(array,per_row_standardization = True):
    if per_row_standardization:
        return array/np.mean(array, axis=1)[:, np.newaxis]
    else:
        return array/np.mean(array)
    
def calculate_softmax(class_distance,normalise_by_row = True):
    if normalise_by_row:
        row_means = np.mean(class_distance, axis=1)
        class_distance =1*  class_distance / row_means[:, np.newaxis]

    soft_max = np.exp(-class_distance)
    soft_max_norm = np.sum(soft_max,axis = 1)
    soft_max = soft_max / soft_max_norm[:, np.newaxis]
    return soft_max

    
