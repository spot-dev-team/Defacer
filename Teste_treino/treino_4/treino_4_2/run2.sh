#!/bin/bash
#SBATCH --job-name=Defacer_treino_4
#SBATCH --account=haslab
#SBATCH --nodelist=aurora06
#SBATCH --partition=rtx4060
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2     # MANTÉM 2: Com 16GB, 4 CPUs causam OOM Kill
#SBATCH --mem=16G             # O teu limite máximo
#SBATCH --time=24:00:00
#SBATCH --output=logs/treino_4_2.log
#SBATCH --error=logs/treino_4_2.err

# 1. Carregar módulos (Versões compatíveis com RTX 4060)
module purge
module load Python/3.11.2
module load CUDA/12.3.0
module load cuDNN/8.9.7.29

# 2. Ir para a pasta do projeto
cd /home/andresousa615/defacer

# 3. Ativar o ambiente virtual (que recriámos no Passo 1)
source venv_defacer/bin/activate

# 4. Debug (Para confirmares no log que a GPU foi apanhada)
echo "--- INÍCIO DO JOB ---"
echo "Data: $(date)"
echo "Nó: $(hostname)"
echo "GPUs Detetadas (nvidia-smi):"
nvidia-smi
echo "GPUs Detetadas (TensorFlow):"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
echo "---------------------"

# 5. Executar
# ATENÇÃO: Confirma que no ficheiro Python tens workers=2 e max_queue_size=2
python train_t4_t2.py
