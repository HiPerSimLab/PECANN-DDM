#!/bin/bash
#SBATCH --job-name=test
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1                # node count
#SBATCH --nodelist=gpu-n61
#SBATCH --gres=gpu:1     
#SBATCH --cpus-per-task=25
#SBATCH --mem=256G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=00-02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --qos=short
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
            

mpirun --oversubscribe --np 25 --bind-to core -map-by socket:PE=1 --report-bindings python -u main.py

