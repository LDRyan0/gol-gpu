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
module use gcc/8.3.0
module use /group/courses0100/software/nvhpc/modulefiles
module load nvhpc/21.9

# Python modules for plotting script
module load python/3.6.3
module load numpy/1.19.0
module load pip/20.2.4
pip install matplotlib

# clean out performance.txt file from previous
rm performance.txt

# add header to performance.txt for user clarity
echo "n,m,steps,cpu_time_ms,gpu_time_ms,speedup,kernel_percent" >> performance.txt 

srun make clean
srun make

# run desired instances of game of life
srun ./bin/game_of_life_cuda 10 10 100
srun ./bin/game_of_life_cuda 100 100 100
srun ./bin/game_of_life_cuda 1000 1000 100
srun ./bin/game_of_life_cuda 10000 10000 100

# generate plots using Python script
python plot.py