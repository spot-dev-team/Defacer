#!/bin/bash
#SBATCH --job-name=Defacer_Deucalion_t1
#SBATCH --account=f202500001hpcvlabepicureg
#SBATCH --partition=normal-a100-40     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32             # Deucalion: 32 CPUs por cada GPU A100
#SBATCH --mem=128G                     
#SBATCH --time=18:00:00                # Tempo máximo para treino longo
#SBATCH --output=logs/treino_deucalion_%j.log
#SBATCH --error=logs/treino_deucalion_%j.err
#SBATCH --gpus=1

# --- CONFIGURAÇÃO DE CAMINHOS ---
PROJECT_ROOT="/projects/F202500001HPCVLABEPICURE/andresousa615/defacer"
export ADNI_DIR="$PROJECT_ROOT/ADNI_Lite_Cluster"
export IXI_DIR="$PROJECT_ROOT/IXI_Cluster"
export MODEL_OUTPUT="model_final_deucalion.h5"

# 1. Carregar módulos 
module purge
module load Miniconda3/23.5.2-0
module load CUDA/12.2.2

# 2. Ativar o ambiente virtual (Sintaxe correta e Caminho Absoluto)
# 2. Ativar o ambiente virtual e exportar variáveis da GPU
eval "$(conda shell.bash hook)"
source activate /projects/F202500001HPCVLABEPICURE/andresousa615/defacer/conda_env

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib

# 3. Ir para a pasta de execução
cd $PROJECT_ROOT/treino_deucalion

# 4. Debug 
echo "--- INÍCIO DO JOB NO DEUCALION ---"
echo "Data: $(date)"
echo "Nó: $(hostname)"
nvidia-smi
python -c "import tensorflow as tf; print('GPUs Detetadas:', tf.config.list_physical_devices('GPU'))"
echo "----------------------------------"

# 5. Executar e medir tempo
start_time=$(date +%s)

python train_deucalion_t1.py

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "----------------------------------"
echo "--- FIM DO JOB NO DEUCALION ---"
echo "Data de Finalização: $(date)"
echo "Tempo Total de Execução: $duration segundos"
echo "----------------------------------"
