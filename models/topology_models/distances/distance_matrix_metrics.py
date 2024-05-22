import numpy as np
def calculate_linear_affinity(distance_matrix):
    dim = distance_matrix.shape[0]
    assert distance_matrix.shape[0] == distance_matrix.shape[1]
    tst_mat = np.eye(dim) * - 4.5 + (np.ones((dim,dim)) - np.eye(dim))
    soft_max = calculate_softmax(distance_matrix)
    return np.sum(tst_mat * soft_max)

def calculate_crossentropy(distance_matrix):
    soft_max = calculate_softmax(distance_matrix)
    cross_entropy = np.log(np.prod(np.diag(soft_max)))
    return cross_entropy

def calculate_variational_ratio(distance_matrix):
    soft_max = calculate_softmax(distance_matrix)
    variational_ratio = np.sum(np.diag(soft_max))/distance_matrix.shape[0]
    return variational_ratio


def calculate_softmax(class_distance):
    soft_max = np.exp(-class_distance)
    soft_max_norm = np.sum(soft_max,axis = 1)
    soft_max = soft_max / soft_max_norm[:, np.newaxis]
    return soft_max

def get_score_of_distances(distance_mat):
    linear_affinity = calculate_linear_affinity(distance_mat)
    cross_entropy = calculate_crossentropy(distance_mat)
    var_ratio = calculate_variational_ratio(distance_mat)
    return {"linear_affinity":linear_affinity,"cross_entropy":cross_entropy,"var_ratio":var_ratio}
