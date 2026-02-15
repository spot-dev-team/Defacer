import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# --- CONFIGURAÇÃO ---
# Confirma os nomes dos teus ficheiros CSV
FILE_AUG = '../treino_2/training_log_final.csv'
FILE_NO_AUG = '../treino_3/training_log_no_aug.csv'

# Verificar se existem
if not os.path.exists(FILE_AUG) or not os.path.exists(FILE_NO_AUG):
    print("ERRO: Faltam ficheiros CSV. Tens de ter o 'training_log_final.csv' e o 'training_log_no_aug.csv' na mesma pasta.")
    # Se não tiveres o antigo, comenta a linha abaixo para gerar só o do novo
    # sys.exit()

# Carregar dados
try:
    df_aug = pd.read_csv(FILE_AUG)
    df_no_aug = pd.read_csv(FILE_NO_AUG)
except:
    print("Erro ao ler os CSVs.")
    sys.exit()

# --- PLOT ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# GRÁFICO 1: TREINO (A prova do Overfitting)
# O modelo sem Augmentation deve subir muito mais rápido (memorização)
ax1.plot(df_no_aug['epoch'], df_no_aug['dice_score'], label='Sem Augmentation (Treino)', color='red', linestyle='--')
ax1.plot(df_aug['epoch'], df_aug['dice_score'], label='Com Augmentation (Treino)', color='green')
ax1.set_title('Dice Score no TREINO (Memorização)')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Dice Score')
ax1.legend()
ax1.grid(True, alpha=0.3)

# GRÁFICO 2: VALIDAÇÃO (A prova da Robustez)
# O modelo com Augmentation deve ser mais estável ou superior a longo prazo
ax2.plot(df_no_aug['epoch'], df_no_aug['val_dice_score'], label='Sem Augmentation (Validação)', color='red', linestyle='--')
ax2.plot(df_aug['epoch'], df_aug['val_dice_score'], label='Com Augmentation (Validação)', color='green')
ax2.set_title('Dice Score na VALIDAÇÃO (Generalização)')
ax2.set_xlabel('Épocas')
ax2.set_ylabel('Dice Score')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle("Impacto do Data Augmentation: Memorização vs. Generalização", fontsize=16)
plt.tight_layout()
plt.savefig('comparacao_aug_vs_noaug.png')
print("✅ Gráfico gerado: comparacao_aug_vs_noaug.png")