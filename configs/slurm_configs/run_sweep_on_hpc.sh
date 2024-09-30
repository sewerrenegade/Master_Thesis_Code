#!/bin/bash
#SBATCH --job-name=wandb-2-agent-job                                # Job name
#SBATCH --output=/home/icb/milad.bassil/logs/wandb_agent_%j.out     # Output file
#SBATCH --error=/home/icb/milad.bassil/logs/wandb_agent_%j.err      # Error file
#SBATCH --time=24:00:00                                             # Maximum runtime (e.g., 24 hours)
#SBATCH --partition=gpu_p                                           # Partition name
#SBATCH --ntasks=1                                                  # Number of tasks (for single-node)
#SBATCH --cpus-per-task=10                                           # Number of CPU cores per task
#SBATCH --mem=64G                                                   # Memory allocation
#SBATCH --gres=gpu:1                                                # Request one GPU (if needed)
#SBATCH --qos=gpu_normal                                            # QOS set
#SBATCH --nodes=1                                                   # number nodes

echo $HOSTNAME
echo "Current directory: $(pwd)"
cd /home/icb/milad.bassil/Desktop/Master_Thesis_Code
echo "Current directory: $(pwd)"
source /home/icb/milad.bassil/miniconda3/etc/profile.d/conda.sh
conda activate masters
echo $CONDA_DEFAULT_ENV
# Run the sweeping agent command
wandb agent milad-research/param_sweep/o94lp36s
