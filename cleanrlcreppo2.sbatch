#!/bin/bash
#SBATCH --job-name=cleanrl_creppo_mujoco
#rrSBATCH -p   mcml-dgx-a100-40x8,mcml-hgx-a100-80x4,mcml-hgx-h100-94x4 
#SBATCH -p  lrz-dgx-1-p100x8,lrz-dgx-1-v100x8,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4,lrz-hgx-h100-94x4
#rrSBATCH --qos=mcml
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#rrSBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue

readonly PROJECT_PATH=$(cd /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95xob2/cleanrl/; pwd)
readonly JOBLOG="$(pwd)/logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.log"

# Job start
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

source ~/.bashrc
mamba activate cleanrl2
cd "$PROJECT_PATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


python -m cleanrl_utils.benchmark \
    --env-ids Hopper-v4  \
    --command "python cleanrl/credal_ppo_continuous_action.py --track --wandb-project-name cleanrlmujoco" \
    --num-seeds 3 \
    --workers 9     &>> ${JOBLOG}

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}