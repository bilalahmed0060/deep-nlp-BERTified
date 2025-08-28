#!/bin/bash
#SBATCH --job-name=train-multitask_classifier
#SBATCH -t 00:20:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p grete:shared              # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                    # take 1 GPU, see https://docs.hpc.gwdg.de/compute_partitions/gpu_partitions/index.html for more options
#SBATCH --mem-per-gpu=8G             # setting the right constraints for the splitted gpu partitions
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=8            # number cores per task
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=TODO@stud.uni-goettingen.de   
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

source activate dnlp # Or whatever you called your environment.

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env 2> /dev/null

# Print out some git info.
module load git
echo -e "\nCurrent Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Latest Commit: $(git rev-parse --short HEAD)"
echo -e "Uncommitted Changes: $(git status --porcelain | wc -l)\n"

# Run the script:
# python -u multitask_classifier.py --use_gpu --local_files_only --option finetune --task sst --hidden_dropout_prob 0.1
# python -u multitask_classifier.py --use_gpu --local_files_only --option finetune --task etpc --hidden_dropout_prob 0.1 --batch_size 32
# python finetune_bart_detection-asl.py --gamma_neg 2.0 --clip 0.1 --logit_adjust_tau 0.5 --use_weighted_sampler --use_gpu
python finetune_bart_detection-dbloss.py --alpha 0.1 --beta 10 --mu 0.9 --lam 2.0 --kappa 0.05 --gamma 0.0 --logit_adjust_tau 0.5 --use_weighted_sampler --use_gpu
