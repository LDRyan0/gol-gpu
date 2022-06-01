#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --time=00:05:00
#SBATCH --export=NONE
#SBATCH --reservation=GS-20880
#SBATCH --account=courses0100

# modules for core functionality
module load pgi/19.7
module load cascadelake 

# Python modules for plotting script
module load python/3.6.3
module load numpy/1.19.0
module load pip/20.2.4
pip install matplotlib

# clean out performance.txt file from previous
# rm performance.txt

# add header to performance.txt for user clarity
# echo "n,m,steps,cpu_time_ms,gpu_time_ms,speedup" >> performance.txt 

srun make clean
srun make

# run desired instances of game of life
srun ./bin/game_of_life_cuda 0 0 0 

# generate plots using Python script
# python plot.py