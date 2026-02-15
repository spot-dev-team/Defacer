import os
import time
import shutil
# Importamos a tua classe atualizada do tools.py
# (Certifica-te que o tools.py na mesma pasta tem a função _find_smart_nose_bottom)
from tools import SpotSegmenter

# --- CONFIGURAÇÃO ---
# A pasta onde estão os teus dados (D: ou C:)
BASE_DIR = r"D:\Tese_BDs\Defacer\segmentados"

def run_reprocessing():
    if not os.path.exists(BASE_DIR):
        print(f"[ERRO] Pasta não encontrada: {BASE_DIR}")
        return

    # Listar pacientes
    subjects = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    total = len(subjects)
    
    print(f"--- REPROCESSAMENTO DE SEGMENTAÇÃO (Update Nariz) ---")
    print(f"Diretório: {BASE_DIR}")
    print(f"Total de exames a verificar: {total}")
    print("-" * 50)

    count_processed = 0
    count_skipped = 0
    count_error = 0

    for i, subj in enumerate(subjects):
        subj_path = os.path.join(BASE_DIR, subj)
        
        # Caminhos de Input (Já existentes)
        mid_dir = os.path.join(subj_path, "mid_output")
        face_mask = os.path.join(mid_dir, "face_mask.nii.gz")
        eyes_raw = os.path.join(mid_dir, "eyes_raw.nii.gz")
        
        # Caminho de Output (Para onde vai a nova máscara)
        final_dir = os.path.join(subj_path, "training_masks")
        final_mask_target = os.path.join(final_dir, "mask_4_classes.nii.gz")

        if os.path.exists(final_mask_target):
            os.remove(final_mask_target)  # Remove a máscara antiga para garantir que é gerada uma nova
            print(f"[{i+1}/{total}] 🗑️ Máscara antiga removida para: {subj}")
            

        # Verificar se temos os ingredientes necessários
        if os.path.exists(face_mask) and os.path.exists(eyes_raw):
            print(f"[{i+1}/{total}] A reprocessar: {subj} ...", end="\r")
            
            try:
                # Criar pasta de destino se não existir (ex: nos casos migrados do C: pode faltar)
                if not os.path.exists(final_dir):
                    os.makedirs(final_dir)

                # --- O CORAÇÃO DO PROCESSO ---
                # Instancia o Segmentador (que agora tem a lógica nova do nariz)
                segmenter = SpotSegmenter(face_mask, eyes_raw, final_dir)
                segmenter.process()
                
                # Gera o ficheiro 'spot_final_combined.nii.gz'
                generated_path = segmenter.save_final_mask()
                
                # Renomeia/Move para o nome padrão 'mask_4_classes.nii.gz'
                if os.path.exists(generated_path):
                    shutil.move(generated_path, final_mask_target)
                    print(f"[{i+1}/{total}] ✅ ATUALIZADO: {subj}           ")
                    count_processed += 1
                else:
                    print(f"[{i+1}/{total}] ❌ ERRO: Ficheiro não gerado para {subj}")
                    count_error += 1

            except Exception as e:
                print(f"\n[{i+1}/{total}] ⚠️ ERRO CRÍTICO em {subj}: {e}")
                count_error += 1
        else:
            # Se não tiver mid_output, ignoramos (provavelmente ainda não foi processado pelo FreeSurfer)
            # print(f"[{i+1}/{total}] SKIP (Sem dados intermédios): {subj}")
            count_skipped += 1

    print("-" * 50)
    print("REPROCESSAMENTO CONCLUÍDO")
    print(f"✅ Exames atualizados com nova máscara: {count_processed}")
    print(f"⏭️ Exames ignorados (sem dados base): {count_skipped}")
    print(f"❌ Erros: {count_error}")
    print("-" * 50)

if __name__ == "__main__":
    run_reprocessing()