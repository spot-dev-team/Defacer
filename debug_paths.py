import os
import sys

print("--- DIAGNÓSTICO DE CAMINHOS ---")
# 1. Onde o terminal está "sentado" (Current Working Directory)
print(f"O teu terminal está em: {os.getcwd()}")

# 2. Onde este script de debug está guardado
print(f"O script de debug está em: {os.path.abspath(__file__)}")

# 3. Verificar se a pasta 'model' existe a partir daqui
model_path = os.path.join(os.getcwd(), 'model_distribution', 'model')
print(f"A procurar pasta 'model' em: {model_path}")

if os.path.exists(model_path):
    print("✅ Pasta 'model' ENCONTRADA!")
    # 4. Listar o que está dentro da pasta model
    print(f"Conteúdo de 'model/': {os.listdir(model_path)}")
else:
    print("❌ Pasta 'model' NÃO ENCONTRADA neste local.")

# 5. Tentar localizar o ficheiro específico
h5_file = os.path.join(model_path, 'model_contour4.h5')
if os.path.exists(h5_file):
    print(f"✅ Ficheiro {h5_file} ENCONTRADO!")
else:
    print(f"❌ Ficheiro {h5_file} DESAPARECIDO.")
print("-------------------------------")