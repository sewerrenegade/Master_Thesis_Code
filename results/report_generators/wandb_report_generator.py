import wandb
import pandas as pd
import re
import numpy as np

def get_split_index(name):
    match = re.search(r'split_(\d+)_of_', run_name)
    if match:
        split_number = int(match.group(1))
        return split_number
    else:
        return -1
    
# Initialize a W&B API client
api = wandb.Api()

# Define your W&B project
entity = "milad-research"  # W&B entity (user or team name)
project = "Final_Results"  # W&B project name


# list_of_run_types = ["og_moor_mani_grayscale","og_moor_mani_dino","og_moor_eucl_grayscale","og_moor_eucl_dino","edge_dist_match_eucl_grayscale",
#                      "edge_dist_match_eucl_dino","baseline_avg_pool","baseline"]
list_of_run_types = ["baseline_s"]
metrics_of_interest = ["accuracy","recall_macro","precision_macro","f1_macro","auroc"]
folds = 4
# Fetch all runs in the project
runs = api.runs(f"{entity}/{project}")

# Define a dictionary to hold your results
results = {run_type:{fold_number:{metric:[] for metric in metrics_of_interest} for fold_number in [i+1 for i in range(folds)]} for run_type in list_of_run_types}
# Iterate through the runs and collect the relevant data
for run in runs:
    # model_name = run.config.get("model_name", "unknown_model")  # Assuming you logged model name in config
    # fold = run.config.get("fold", "unknown_fold")  # Assuming fold info is logged
    

    run_name = run.name
    run_fold_number = get_split_index(run_name)
    run_type = next((exp for exp in list_of_run_types if exp in run_name),"Unknown Type")

    for metric in metrics_of_interest:
        metric_value = run.summary.get(metric)  # Replace 'accuracy' with the metric you want to analyze
        if metric_value is not None and run_type!="Unknown Type":
            results[run_type][run_fold_number][metric].append(metric_value)

summary = []
across_split_metrics = []

for run_type, splits in results.items():
    temp_across_split_metrics = {metric:[] for metric in metrics_of_interest}
    for split_n, metrics in splits.items():
        for metric, metric_values in metrics.items():
            if len(metric_values) != 0:
                summary.append({
                    "run_type": run_type,
                    "fold": split_n,
                    "metric": metric,
                    # "mean": np.mean(metric_values),
                    # "std": np.std(metric_values),
                    "mean_std_string": f"{np.mean(metric_values):.3f} +/- {np.std(metric_values):.3f}"
                })
            temp_across_split_metrics[metric].extend(metric_values)

    for metric_name, metric_values_across_splits in temp_across_split_metrics.items():
            
        across_split_metrics.append({
                        "run_type": run_type,
                        "metric": metric_name,
                        # "mean": np.mean(metric_values_across_splits),
                        # "std": np.std(metric_values_across_splits),
                        "mean_std_string": f"{np.mean(metric_values_across_splits):.3f} +/- {np.std(metric_values_across_splits):.3f}"
                    })
        summary.append({
                        "run_type": run_type,
                        "metric": metric_name,
                        "fold": "All",
                        # "mean": np.mean(metric_values_across_splits),
                        # "std": np.std(metric_values_across_splits),
                        "mean_std_string": f"{np.mean(metric_values_across_splits):.3f} +/- {np.std(metric_values_across_splits):.3f}"
                    })
# Create a Pandas DataFrame
df =  pd.DataFrame(summary)#cross_split_metrics)

pivot_df = df.pivot(index="metric", columns="fold", values="mean_std_string")

# Optionally, rename columns for clarity
pivot_df.columns = [f"split_{col}" for col in pivot_df.columns]

# Reset index for clean CSV output
pivot_df.reset_index(inplace=True)
# Print the DataFrame
print(df)

# Optionally, save the DataFrame to a CSV file
pivot_df.to_csv("model_fold_performance.csv", index=False)

latex_code = pivot_df.to_latex(index=False)

# Save to a .tex file
with open("table.tex", "w") as f:
    f.write(latex_code)