import torch
import torch.nn.functional as F

def calculate_recall(predictions, ground_truth, num_classes):
    total_true_positives = 0
    total_ground_truth_positives = 0
    recalls = []

    for class_idx in range(num_classes):
        true_positives = torch.sum((predictions == class_idx) & (ground_truth == class_idx))
        false_negatives = torch.sum((predictions != class_idx) & (ground_truth == class_idx))

        recall = true_positives / true_positives + false_negatives
        recalls.append(recall.item())

        total_true_positives += true_positives
        total_ground_truth_positives += (true_positives + false_negatives)

    overall_recall = total_true_positives / (total_true_positives + false_negatives + 1e-7)
    return overall_recall

def calculate_precision(predictions, ground_truth, num_classes):
    total_true_positives = 0
    total_predicted_positives = 0
    precisions = []

    for class_idx in range(num_classes):
        true_positives = torch.sum((predictions == class_idx) & (ground_truth == class_idx))
        false_positives = torch.sum((predictions == class_idx) & (ground_truth != class_idx))
        
        precision = true_positives / true_positives + false_positives
        precisions.append(precision.item())

        total_true_positives += true_positives
        total_predicted_positives += (true_positives + false_positives)
        
    overall_precision = total_true_positives / (total_predicted_positives + 1e-7)
    return overall_precision.item()

def calculate_sensitivity(predictions, ground_truth, num_classes):
    total_true_positives = 0
    total_ground_truth_positives = 0
    sensitivities = []

    for class_idx in range(num_classes):
        true_positives = torch.sum((predictions == class_idx) & (ground_truth == class_idx))
        false_negatives = torch.sum((predictions != class_idx) & (ground_truth == class_idx))

        sensitivity = true_positives / (true_positives + false_negatives + 1e-7)
        sensitivities.append(sensitivity.item())

        total_true_positives += true_positives
        total_ground_truth_positives += (true_positives + false_negatives)
    
    overall_sensitivity = total_true_positives / (total_ground_truth_positives + 1e-7)
    return overall_sensitivity.item()
