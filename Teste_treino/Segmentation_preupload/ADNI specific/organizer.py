import os
import shutil
from pathlib import Path

# --- CONFIGURAÇÃO ---
# Pasta onde tens tudo misturado agora
SOURCE_DIR = r"C:\Tese\Datasets\Defacer\ADNI_Nifti_Single_Folder"

# Nova pasta onde vamos organizar tudo (O script cria-a se não existir)
DEST_DIR = r"C:\Tese\Datasets\Defacer\ADNI_Structured"

def organize_dataset():
    # 1. Criar diretório de destino
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"[Organizador] Pasta criada: {DEST_DIR}")

    # 2. Listar ficheiros
    files = [f for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f))]
    
    count = 0
    print(f"[Organizador] Encontrados {len(files)} ficheiros na origem. A filtrar...")

    for filename in files:
        # A. Filtros: Queremos apenas .nii.gz
        if not filename.endswith(".nii.gz"):
            continue
            
        # B. Filtros de Segurança: Ignorar máscaras antigas ou ficheiros de teste
        # Se os teus exames originais tiverem nomes "limpos", isto ajuda a não copiar lixo
        if "_mask" in filename or "_seg" in filename or "spot_" in filename:
            print(f" -> Ignorado (parece ficheiro derivado): {filename}")
            continue

        # C. Lógica de Criação da Pasta
        # O nome da pasta será o nome do ficheiro sem a extensão .nii.gz
        folder_name = filename.replace(".nii.gz", "")
        
        # Caminho da nova "casa" deste exame
        subject_folder = os.path.join(DEST_DIR, folder_name)
        
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)

        # D. Copiar o Ficheiro
        src_path = os.path.join(SOURCE_DIR, filename)
        
        # Vamos manter o nome original ou renomear para 'raw.nii.gz'?
        # Manter o original é mais seguro para rastreabilidade nesta fase.
        dst_path = os.path.join(subject_folder, filename) 

        print(f" -> A copiar: {filename} ...")
        shutil.copy2(src_path, dst_path)
        count += 1

    print("-" * 30)
    print(f"[Sucesso] {count} exames organizados em: {DEST_DIR}")

if __name__ == "__main__":
    organize_dataset()