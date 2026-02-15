import os
import shutil
import time

# --- CONFIGURAÇÃO ---
# Onde estão os dados misturados (Disco C cheio)
SOURCE_DIR = r"C:\Tese\IXI_Structured"

# Para onde vão os dados (Disco D limpo)
DEST_DIR = r"D:\Tese_BDs\Defacer\IXI_Structured"

def migrate_successful_segmentations():
    if not os.path.exists(SOURCE_DIR):
        print(f"ERRO: Diretório de origem não existe: {SOURCE_DIR}")
        return
    if not os.path.exists(DEST_DIR):
        print(f"ERRO: Diretório de destino não existe: {DEST_DIR}")
        return

    # Listar todos os pacientes no C:
    subjects = [f for f in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, f))]
    total = len(subjects)
    
    print(f"--- A INICIAR MIGRAÇÃO ---")
    print(f"Origem: {SOURCE_DIR}")
    print(f"Destino: {DEST_DIR}")
    print(f"Total de pastas a verificar: {total}")
    print("-" * 50)

    count_moved = 0
    count_skipped = 0
    bytes_transferred = 0

    for i, subj in enumerate(subjects):
        src_subj_path = os.path.join(SOURCE_DIR, subj)
        dst_subj_path = os.path.join(DEST_DIR, subj)

        # 1. VERIFICAR SE O PROCESSO TERMINOU COM SUCESSO
        # O critério de sucesso é existir o ficheiro final: training_masks/mask_4_classes.nii.gz
        final_mask = os.path.join(src_subj_path, "training_masks", "mask_4_classes.nii.gz")
        
        if not os.path.exists(final_mask):
            # print(f"[{i+1}/{total}] SKIP {subj}: Não tem segmentação completa.")
            count_skipped += 1
            continue

        print(f"[{i+1}/{total}] MIGRAÇÃO {subj} (Sucesso detetado)...")

        # 2. LISTA DE FICHEIROS A COPIAR
        # Vamos definir exatamente o que queremos levar para não encher o D com lixo
        files_to_copy = [
            ("training_masks", "mask_4_classes.nii.gz"),
            ("mid_output", "eyes_raw.nii.gz"),
            ("mid_output", "face_mask.nii.gz"),
            ("mid_output", "mideface_anon_temp.nii.gz")
        ]

        # Garante que a pasta de destino existe (se não existir, cria)
        if not os.path.exists(dst_subj_path):
            os.makedirs(dst_subj_path)

        for folder, filename in files_to_copy:
            src_file = os.path.join(src_subj_path, folder, filename)
            
            # Destino: D:\...\Sujeito\mid_output\ficheiro
            dst_folder_path = os.path.join(dst_subj_path, folder)
            dst_file = os.path.join(dst_folder_path, filename)

            if os.path.exists(src_file):
                # Criar subpasta (mid_output ou training_masks) no destino se não existir
                if not os.path.exists(dst_folder_path):
                    os.makedirs(dst_folder_path)
                
                # Copiar
                try:
                    shutil.copy2(src_file, dst_file)
                    bytes_transferred += os.path.getsize(src_file)
                except Exception as e:
                    print(f"   -> ERRO ao copiar {filename}: {e}")
            else:
                # Se for um ficheiro opcional, ok. Se for a mask final, é estranho (já verificámos antes)
                if filename == "mask_4_classes.nii.gz":
                    print(f"   -> AVISO CRÍTICO: Máscara desapareceu durante a cópia!")

        count_moved += 1

    print("-" * 50)
    print("MIGRAÇÃO CONCLUÍDA")
    print(f"✅ Pacientes migrados com sucesso: {count_moved}")
    print(f"⏭️ Pacientes incompletos ignorados: {count_skipped}")
    print(f"💾 Espaço transferido aprox: {bytes_transferred / (1024*1024):.2f} MB")
    print("-" * 50)
    print("DICA: Verifica se está tudo bem no disco D. Se sim, podes apagar o 'IXI_Structured' do disco C para libertar espaço.")

if __name__ == "__main__":
    migrate_successful_segmentations()