import os
import nibabel as nib
import numpy as np
import shutil

# --- CONFIGURAÇÃO ---
# Pasta onde estão os ficheiros originais misturados
RAW_IXI_FOLDER = r"D:\Tese - BDs\IXI-T1" 

# Pasta de destino organizada
DEST_DIR = r"D:\Tese_BDs\Defacer\IXI_Structured"

def process_ixi_dataset():
    if not os.path.exists(RAW_IXI_FOLDER):
        print(f"ERRO: A pasta de origem não existe: {RAW_IXI_FOLDER}")
        return

    # Criar pasta de destino
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    files = [f for f in os.listdir(RAW_IXI_FOLDER) if f.endswith('.nii') or f.endswith('.nii.gz')]
    print(f"--- A INICIAR PROCESSAMENTO DE {len(files)} EXAMES ---\n")

    count_ok = 0
    count_skipped = 0
    count_error = 0

    for i, f in enumerate(files):
        file_id = f.replace('.nii.gz', '').replace('.nii', '')
        
        # Ignorar ficheiros de sistema
        if f.startswith('.'): continue

        subject_dir = os.path.join(DEST_DIR, file_id)
        final_path = os.path.join(subject_dir, "raw.nii.gz")
        
        # Se o ficheiro final já existe, saltamos para poupar tempo
        if os.path.exists(final_path):
            # print(f"[{i+1}] Saltando {file_id} (Já existe)")
            continue

        try:
            src_path = os.path.join(RAW_IXI_FOLDER, f)
            img = nib.load(src_path)
            
            # --- 1. VERIFICAÇÃO DE COMPATIBILIDADE (O FILTRO) ---
            header = img.header
            zooms = header.get_zooms()
            
            # Pega apenas nas 3 primeiras dimensões (X, Y, Z)
            x_res, y_res, z_res = zooms[:3]
            
            # Critério: Entre 0.8 e 1.3 mm em todos os eixos
            is_compatible = (0.8 <= x_res <= 1.3) and \
                            (0.8 <= y_res <= 1.3) and \
                            (0.8 <= z_res <= 1.3)

            if not is_compatible:
                print(f"[{i+1}] ❌ IGNORADO {file_id}: Voxels fora do padrão ({x_res:.2f}, {y_res:.2f}, {z_res:.2f})")
                count_skipped += 1
                continue # Salta para o próximo ficheiro
            
            # --- 2. REORIENTAÇÃO E SALVAMENTO ---
            # Se chegou aqui, é compatível. Vamos processar.
            
            if not os.path.exists(subject_dir):
                os.makedirs(subject_dir)

            # Força orientação RAS (Canonical)
            img_canonical = nib.as_closest_canonical(img)
            
            # Guardar
            nib.save(img_canonical, final_path)
            
            print(f"[{i+1}] ✅ PROCESSADO {file_id}: ({x_res:.2f}, {y_res:.2f}, {z_res:.2f}) -> RAS")
            count_ok += 1

        except Exception as e:
            print(f"[{i+1}] ⚠️ ERRO CRÍTICO em {f}: {e}")
            count_error += 1
            if os.path.exists(subject_dir) and not os.listdir(subject_dir):
                os.rmdir(subject_dir)

    print("\n" + "="*40)
    print("RESUMO FINAL:")
    print(f"✅ Processados com sucesso: {count_ok}")
    print(f"❌ Ignorados (Incompatíveis): {count_skipped}")
    print(f"⚠️ Erros de Leitura: {count_error}")
    print(f"📂 Destino: {DEST_DIR}")
    print("="*40)

if __name__ == "__main__":
    process_ixi_dataset()