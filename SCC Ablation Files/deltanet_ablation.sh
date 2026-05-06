#!/bin/bash -l

# Set SCC project
#$ -P ec523

# Give a name to my job
#$ -N deltanet

#$ -pe omp 4

#$ -l h_rt=10:00:00

#$ -l gpus=1

#$ -l gpu_c=8.0

#$ -l gpu_type="!RTXP6000"

#$ -M mlwe@bu.edu
#$ -m beas

module load python3/3.10.12 cuda/12.2
source /projectnb/ec523/projects/.venv/bin/activate

cd /projectnb/ec523/projects/proj_denoise_speech

python ablation_deltanet.py