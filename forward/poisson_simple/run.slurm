#!/bin/bash
#SBATCH --job-name=test
#SBATCH --cluster=smp
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=25      # total number of tasks per node
#SBATCH --time=00-10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
crc-job-stats

module purge

TF_CPP_MIN_LOG_LEVEL=3

#module load horovod/python3.8_pytorch
module load python/3.7.0 venv/wrap

module load gcc/8.2.0 
module load openmpi/4.0.3 
module load nccl/2.8.4 
module load cuda/11.8 

workon torch_cuda

# Check if mpi4py is installed, if not install it
python -m pip show mpi4py > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing mpi4py..."
    python -m pip install mpi4py
else
    echo "mpi4py is already installed."
fi
            

mpirun -n 25 python -u main.py

