import numpy as np
import concurrent.futures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score,cross_val_predict
import umap
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import random
from PIL import Image
from collections import defaultdict
import io
from distance_functions.distance_function_metrics.abstract_metrics import Metric
from distance_functions.functions.basic_distance_functions import EuclideanDistance
from results.metrics_descriptor import MetricsDescriptor
from tqdm import tqdm

class DistanceMatrixMetricCalculator(Metric):
    def __init__(self, live_descriptor:MetricsDescriptor,num_of_workers_for_distance_matrix_calc:int= 8) -> None:
        self.name =  "distance_matrix_metrics"
        self.output_metrics_format = {
            "intra_to_inter_class_distance_overall_ratio": "scalar",
            "intra_to_inter_class_distance_overall_per_class_ratio":"dict-scalar",
            "intra_inter_class_distance_matrix_mean":"matrix",
            "intra_inter_class_distance_matrix_std":"matrix",
            "triplet_loss": "scalar",
            "silhouette_score": "scalar",
            "loocv_knn_acc": "scalar",
            "loocv_knn_acc_std": "scalar",
            "loocv_knn_report": "report",
            "loocv_confusion_matrix": "matrix",
            "knn_acc": "scalar",
            "knn_acc_std": "scalar",
            "knn_report": "report",
            "knn_confusion_matrix": "matrix"
        }
        self.num_of_workers = num_of_workers_for_distance_matrix_calc
        self.dataset = live_descriptor.dataset
        self.per_class_samples = live_descriptor.per_class_samples
        self.distance_function = live_descriptor.distance_function
        self.distance_matrix, self.labels, self.per_class_indicies = None,None,None
        
    def calculate_metric(self):
        return self.calculate_distance_matrix_metrics()
    
    def get_settings(self):
        return {}
    
    def calculate_distance_matrix_metrics(self):
        if self.distance_matrix is None:
            self.distance_matrix, self.labels, self.per_class_indicies = self.calculate_n_sample_balanced_distance_matrix()
        overall_ratio, class_wise_ratios = self.average_inter_intra_class_distance_ratio(self.distance_matrix, self.labels)
        intra_inter_class_distance_matrix_mean, intra_inter_class_distance_matrix_std = self.average_inter_intra_class_distance_matrix(self.distance_matrix,self.labels)
        triplet_loss = self.evaluate_triplet_loss(self.distance_matrix, self.labels)
        silhouette_score = self.compute_silhouette_score_from_distance_matrix(self.distance_matrix, self.labels)
        loocv_knn_acc, loocv_knn_acc_std, loocv_knn_report, loocv_confusion_matrix = self.knn_loocv_accuracy(self.distance_matrix, self.labels)
        knn_acc, knn_acc_std, knn_report, knn_confusion_matrix = self.evaluate_knn_classifier_from_distance_matrix(self.distance_matrix, self.labels)

        metrics_dict = {
            "intra_to_inter_class_distance_overall_ratio": overall_ratio,
            "intra_to_inter_class_distance_overall_per_class_ratio":class_wise_ratios,
            "intra_inter_class_distance_matrix_mean":intra_inter_class_distance_matrix_mean,
            "intra_inter_class_distance_matrix_std":intra_inter_class_distance_matrix_std,
            "triplet_loss": triplet_loss,
            "silhouette_score": silhouette_score,
            "loocv_knn_acc": loocv_knn_acc,
            "loocv_knn_acc_std": loocv_knn_acc_std,
            "loocv_knn_report": loocv_knn_report,
            "loocv_confusion_matrix": loocv_confusion_matrix,
            "knn_acc": knn_acc,
            "knn_acc_std": knn_acc_std,
            "knn_report": knn_report,
            "knn_confusion_matrix": knn_confusion_matrix
        }

        return metrics_dict


    def calculate_n_sample_balanced_distance_matrix(self, show_progress=True):
        if self.distance_function is None:
            self.distance_function = EuclideanDistance()
        data = self.dataset.get_random_instances_from_all_classes(self.per_class_samples)
        list_data_shuffled, labels_shuffled = self.get_shuffled_list_data_from_class_indexed(data)
        n_samples = len(list_data_shuffled)

        distance_matrix = np.zeros((n_samples, n_samples))
        tmp_dst = self.distance_function
        
        if show_progress:
            progress_bar = tqdm(total=(n_samples * (n_samples - 1)) // 2, desc="Calculating distances")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_of_workers) as executor:
            futures = {}
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    futures[(i, j)] = executor.submit(tmp_dst, [list_data_shuffled[i], list_data_shuffled[j]])
            for (i, j), future in futures.items():
                distance = future.result()
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
                if show_progress:
                    progress_bar.update(1)
            
            if show_progress:
                progress_bar.close()

            per_class_indices = {key: [] for key in list(data.keys())}
            for label_index, label in enumerate(labels_shuffled):
                per_class_indices[label].append(label_index)
                
        return distance_matrix, np.array(labels_shuffled), per_class_indices

    def triplet_loss(self,anchor_idx, positive_idx, negative_idx, distance_matrix, margin=1.0):
        pos_dist = distance_matrix[anchor_idx, positive_idx]
        neg_dist = distance_matrix[anchor_idx, negative_idx]
        return max(0, pos_dist - neg_dist + margin)

    def evaluate_triplet_loss(self,distance_matrix, labels, margin=1.0):
        unique_labels = np.unique(labels)
        triplet_losses = []
        for label in unique_labels:
            class_indices = np.where(labels == label)[0]
            if len(class_indices) < 2:
                continue
            non_class_indices = np.where(labels != label)[0]
            for anchor_idx in class_indices:
                positive_idx = np.random.choice(class_indices[class_indices != anchor_idx])
                negative_idx = np.random.choice(non_class_indices)
                triplet_losses.append(self.triplet_loss(anchor_idx, positive_idx, negative_idx,distance_matrix, margin = margin))
        return np.mean(triplet_losses)

    def compute_silhouette_score_from_distance_matrix(self,distance_matrix, labels):
        return silhouette_score(distance_matrix, labels, metric='precomputed')

    def  visualize_embeddings_from_distance_matrix(self,distance_matrix, labels, method='umap'):
        if method == 'tsne':
            embedding = TSNE(metric='precomputed').fit_transform(distance_matrix)
        elif method == 'umap':
            embedding = umap.UMAP(metric='precomputed').fit_transform(distance_matrix)
        else:
            raise ValueError("Unsupported method. Use 'tsne' or 'umap'.")
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=100)
        plt.colorbar()
        
        # Convert plot to PIL image
        plt.tight_layout()
        fig_buf = io.BytesIO()
        plt.savefig(fig_buf, format='png')
        fig_buf.seek(0)
        plt.close()
        
        # Convert buffer to PIL image
        pil_image = Image.open(fig_buf)
        return pil_image


    #Solve info leakage problem
    def knn_loocv_accuracy(self,distance_matrix, labels, k=3):
        loo = LeaveOneOut()
        knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
        
        # Perform LOOCV and collect predictions
        predictions = cross_val_predict(knn, distance_matrix, labels, cv=loo)
        
        # Calculate accuracy and std
        accuracy = np.mean(predictions == labels)
        std = np.std(predictions == labels)
        
        # Print classification report
        report = classification_report(labels, predictions)
        conf_matrix = confusion_matrix(labels, predictions)
        
        return accuracy, std, report, conf_matrix

    def evaluate_knn_classifier_from_distance_matrix(self,distance_matrix, labels, k=3):
        knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
        knn.fit(distance_matrix, labels)
        predictions = knn.predict(distance_matrix)
        report = classification_report(labels, predictions)
        accuracy = np.mean(predictions == labels)
        accuracy_std = np.std(predictions == labels)
        conf_matrix = confusion_matrix(labels, predictions)
        return accuracy,accuracy_std,report,conf_matrix
        
        
    def get_shuffled_list_data_from_class_indexed(self,data):
        list_data = []
        labels = []
        for key,value in data.items():
            list_data.extend(value)
            labels.extend([key]*len(value))
        combined = list(zip(list_data, labels))
        random.shuffle(combined)
        list_data_shuffled, labels_shuffled = zip(*combined)
        list_data_shuffled = list(list_data_shuffled)
        labels_shuffled = list(labels_shuffled)
        return list_data_shuffled,labels_shuffled
        
        
    def average_inter_intra_class_distance_ratio(self,distance_matrix, labels):
        n = len(labels)
        classes = np.unique(labels)
        class_wise_ratios = {}
        
        interclass_distances = []
        intraclass_distances = defaultdict(list)
        
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != labels[j]:
                    interclass_distances.append(distance_matrix[i, j])
                else:
                    intraclass_distances[labels[i]].append(distance_matrix[i, j])
        
        if len(interclass_distances) == 0:
            overall_ratio = float('inf')  # No interclass distances available
        else:
            avg_interclass_distance = np.mean(interclass_distances)
            avg_intraclass_distance = np.mean([d for distances in intraclass_distances.values() for d in distances])
            overall_ratio = avg_intraclass_distance / avg_interclass_distance
        
        for cls in classes:
            if len(intraclass_distances[cls]) == 0:
                class_wise_ratios[cls] = float('inf')  # No intraclass distances for this class
            else:
                avg_intraclass_distance_cls = np.mean(intraclass_distances[cls])
                if len(interclass_distances) == 0:
                    class_wise_ratios[cls] = float('inf')
                else:
                    class_wise_ratios[cls] = avg_intraclass_distance_cls / avg_interclass_distance
        
        return overall_ratio, class_wise_ratios    

    def average_inter_intra_class_distance_matrix(self, distance_matrix, labels):
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        distance_matrix_mean = np.zeros((n_classes, n_classes))
        distance_matrix_std = np.zeros((n_classes, n_classes))
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        
        for i in range(n_classes):
            class_i = unique_labels[i]
            
            for j in range(n_classes):
                class_j = unique_labels[j]
                
                if i == j:  # Intraclass distance
                    intraclass_distances = []
                    indices_i = np.where(labels == class_i)[0]
                    
                    for k in range(len(indices_i)):
                        for l in range(k + 1, len(indices_i)):
                            intraclass_distances.append(distance_matrix[indices_i[k], indices_i[l]])
                    
                    if len(intraclass_distances) > 0:
                        avg_intraclass_distance = np.mean(intraclass_distances)
                        std_intraclass_distance = np.std(intraclass_distances)
                    else:
                        avg_intraclass_distance = 0
                        std_intraclass_distance = 0
                    
                    distance_matrix_mean[i, j] = avg_intraclass_distance
                    distance_matrix_std[i, j] = std_intraclass_distance
                
                else:  # Interclass distance
                    interclass_distances = []
                    indices_i = np.where(labels == class_i)[0]
                    indices_j = np.where(labels == class_j)[0]
                    
                    for k in indices_i:
                        for l in indices_j:
                            interclass_distances.append(distance_matrix[k, l])
                    
                    if len(interclass_distances) > 0:
                        avg_interclass_distance = np.mean(interclass_distances)
                        std_interclass_distance = np.std(interclass_distances)
                    else:
                        avg_interclass_distance = float('inf')
                        std_interclass_distance = 0
                    
                    distance_matrix_mean[i, j] = avg_interclass_distance
                    distance_matrix_std[i, j] = std_interclass_distance
        
        return distance_matrix_mean, distance_matrix_std


    # def calculate_linear_affinity(distance_matrix):
    #     nb_classes = distance_matrix.shape[0]
    #     nb_off_diagonal_to_diag_ratio = nb_classes - 1
    #     assert distance_matrix.shape[0] == distance_matrix.shape[1]
    #     tst_mat = - np.eye(nb_classes)  + (np.ones((nb_classes,nb_classes)) - np.eye(nb_classes))/nb_off_diagonal_to_diag_ratio
    #     #soft_max = calculate_softmax(distance_matrix)
    #     row_means = np.mean(distance_matrix, axis=1)
    #     class_distance =1*  distance_matrix / row_means[:, np.newaxis]
    #     return np.sum(tst_mat * class_distance) + nb_classes

    # def calculate_crossentropy(distance_matrix):
    #     soft_max = calculate_softmax(distance_matrix)
    #     cross_entropy = -np.log(np.prod(np.diag(soft_max)))
    #     return cross_entropy

    # def calculate_variational_ratio(distance_matrix):
    #     soft_max = calculate_softmax(distance_matrix)
    #     variational_ratio = np.sum(1-np.diag(soft_max))/distance_matrix.shape[0]
    #     return variational_ratio

    # def calculate_k_neighbors_class_dissimilarity(data_used,nb_per_class_sample_points=5,nb_neighbours = 5):
    #     data,lables = flatten_dict_and_create_labels(data_used)
    #     scores = []
    #     k_neighbors_name =f"k_{nb_neighbours}_neighbourhood_dissimilarity"
    #     for key,value in data_used.items():
    #         for target_point in value[:nb_per_class_sample_points]:
    #             closest_indicies = calculate_the_index_of_k_closest_points(data,target_point,nb_neighbours)
    #             closest_points_labels = [lables[i] for i in closest_indicies]
    #             scores.append(closest_points_labels.count(key)/len(closest_points_labels))
    #     return 1 - np.mean(np.array(scores)),k_neighbors_name

    # def calculate_all_distance_scores(distance_mat,data_used):
    #     distance_mat = standardize_array_mean(distance_mat)
    #     linear_affinity = calculate_linear_affinity(distance_mat)
    #     cross_entropy = calculate_crossentropy(distance_mat)
    #     var_ratio = calculate_variational_ratio(distance_mat)
    #     k_neighbors_average_class_dissimilarity,k_neighbors_name = calculate_k_neighbors_class_dissimilarity(data_used,nb_neighbours=10)#needs to incorporate distance function TODO  
    #     return {k_neighbors_name:k_neighbors_average_class_dissimilarity,"linear_affinity":linear_affinity,"cross_entropy":cross_entropy,"var_ratio":var_ratio}

    # #retruns the indicies of top k closest neighbour points
    # def calculate_the_index_of_k_closest_points(data,target_point,nb_neighbours = 5):
    #     #other_points = np.delete(data, target_index, axis=0)
    #     distances = np.linalg.norm(data - target_point, axis=1)
    #     closest_indices = np.argsort(distances)[:nb_neighbours+1]
    #     closest_indices = list(closest_indices)
    #     if distances[closest_indices[0]] == 0:
    #         closest_indices.remove(closest_indices[0])
    #     else:
    #         print("very weird behaviour")
    #         closest_indices = closest_indices[:-1]
    #     return closest_indices


    # def flatten_dict_and_create_labels(data):
    #     labels = []
    #     flat_data = []
    #     for key,val in data.items():
    #         n = len(val)
    #         labels.extend([key]*n)
    #         flat_data.extend(val)
    #     return np.array(flat_data),labels

    # #makes it so the distribution of ALL distances to be 0 mean and have a variance of 1
    # def center_at_zero_and_normalise(array):
    #     avg = np.mean(array)
    #     std = np.std(array)
    #     return (array-avg)/std

    # #if true makes that average distance between a class and all others (including itself) equal to 1
    # #if false makes the average distance between all classes equal to 1
    # def standardize_array_mean(array,per_row_standardization = True):
    #     if per_row_standardization:
    #         return array/np.mean(array, axis=1)[:, np.newaxis]
    #     else:
    #         return array/np.mean(array)
        
    # def calculate_softmax(class_distance,normalise_by_row = True):
    #     if normalise_by_row:
    #         row_means = np.mean(class_distance, axis=1)
    #         class_distance =1*  class_distance / row_means[:, np.newaxis]

    #     soft_max = np.exp(-class_distance)
    #     soft_max_norm = np.sum(soft_max,axis = 1)
    #     soft_max = soft_max / soft_max_norm[:, np.newaxis]
    #     return soft_max

    
