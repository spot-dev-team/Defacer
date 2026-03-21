#!/bin/bash

#SBATCH --job-name=Defacer_Baseline

#SBATCH --account=haslab

#SBATCH --partition=rtx4060



#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=16G                

#SBATCH --time=24:00:00

#SBATCH --output=logs/spot_%j.log

#SBATCH --error=logs/spot_%j.err



# 1. Carregar módulos

module purge

module load Python/3.10.4-GCCcore-11.3.0

module load CUDA/11.3.1

module load cuDNN/8.2.1.32-CUDA-11.3.1



# 2. Ir para a pasta do projeto

cd /home/andresousa615/defacer



# 3. Ativar ambiente virtual

source venv_defacer/bin/activate



# 4. Debug

echo "Job iniciado em: $(date)"

echo "Nó: $(hostname)"

nvidia-smi



# 5. Executar

python train_adni_keras.py
