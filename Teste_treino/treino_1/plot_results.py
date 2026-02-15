import pandas as pd
import matplotlib.pyplot as plt
import sys

# Tenta carregar o CSV
try:
    data = pd.read_csv('training_log.csv')
except FileNotFoundError:
    print("ERRO: Não encontro o ficheiro 'training_log.csv'.")
    sys.exit()

# Configurar o gráfico
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Dice Score (Quanto maior, melhor)
ax1.plot(data['epoch'], data['dice_score'], label='Treino (Dice)', color='blue')
ax1.plot(data['epoch'], data['val_dice_score'], label='Validação (Dice)', color='orange', linestyle='--')
ax1.set_title('Evolução do Dice Score (Precisão)')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Score (0 a 1)')
ax1.legend()
ax1.grid(True)

# Gráfico 2: Loss (Quanto menor, melhor)
ax2.plot(data['epoch'], data['loss'], label='Treino (Loss)', color='blue')
ax2.plot(data['epoch'], data['val_loss'], label='Validação (Loss)', color='orange', linestyle='--')
ax2.set_title('Evolução da Loss (Erro)')
ax2.set_xlabel('Épocas')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

# Guardar imagem
plt.savefig('resultado_treino.png')
print("✅ Gráfico gerado com sucesso: resultado_treino.png")