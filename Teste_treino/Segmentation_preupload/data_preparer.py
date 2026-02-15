import os
import subprocess
import time
# Apenas importamos o Segmenter, não precisamos do Defacer
from tools import SpotSegmenter



#segmentar dados para treinar o defacer

# --- CONFIGURAÇÃO ---
# A pasta onde o script de conversão guardou os NIfTIs limpos
BASE_DIR = r"D:\Tese_BDs\Defacer\ADNI_Structured"

# Configurações do WSL (Iguais ao maestro.py)
WSL_DISTRO_NAME = "Ubuntu-22.04"
SCRIPT_PATH_LINUX = "~/step1_mideface.sh"

# --- FUNÇÕES AUXILIARES ---
def to_wsl_path(win_path):
    """Converte C:\Pasta para /mnt/c/Pasta"""
    abs_path = os.path.abspath(win_path)
    drive, rest = os.path.splitdrive(abs_path)
    drive_letter = drive.replace(':', '').lower()
    wsl_path = f"/mnt/{drive_letter}{rest.replace(os.sep, '/')}"
    return wsl_path

# --- PIPELINE DE PREPARAÇÃO (SEGMENTAÇÃO APENAS) ---
def run_training_prep():
    if not os.path.exists(BASE_DIR):
        print(f"[ERRO CRÍTICO] Pasta de dados não encontrada: {BASE_DIR}")
        return

    subjects = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    subjects.sort()
    total = len(subjects)
    
    print(f"[DATA PREP] A gerar máscaras de treino para {total} exames...")
    print(f"[DATA PREP] Distro WSL: {WSL_DISTRO_NAME}")
    
    success_count = 0
    
    for i, subj in enumerate(subjects):
        subj_path = os.path.join(BASE_DIR, subj)
        print(f"\n[{i+1}/{total}] Processando: {subj}")
        
        # 1. Encontrar o ficheiro raw (criado pelo conversor)
        # Procuramos especificamente 'raw.nii.gz' para sermos consistentes
        raw_file = os.path.join(subj_path, "raw.nii.gz")
        
        if not os.path.exists(raw_file):
            # Fallback caso o nome seja diferente, mas tenta achar um nifti
            candidates = [f for f in os.listdir(subj_path) if f.endswith(".nii.gz") and "mask" not in f]
            if candidates:
                raw_file = os.path.join(subj_path, candidates[0])
            else:
                print(" -> [SKIP] Nenhum NIfTI raw encontrado.")
                continue
        
        mid_output_dir = os.path.join(subj_path, "mid_output")
        final_output_dir = os.path.join(subj_path, "spot_masks") # Mudei nome para ser claro
        
        # ---------------------------------------------------------
        # PASSO 1: GERAÇÃO INICIAL (Bash via WSL)
        # ---------------------------------------------------------
        face_mask = os.path.join(mid_output_dir, "face_mask.nii.gz")
        eyes_raw = os.path.join(mid_output_dir, "eyes_raw.nii.gz")
        
        if not (os.path.exists(face_mask) and os.path.exists(eyes_raw)):
            print(" -> Passo 1: A gerar segmentações base (MiDeFace)...")
            
            wsl_raw = to_wsl_path(raw_file)
            wsl_out = to_wsl_path(mid_output_dir)
            
            bash_cmd = f"{SCRIPT_PATH_LINUX} \"{wsl_raw}\" \"{wsl_out}\""
            full_cmd = ["wsl", "-d", WSL_DISTRO_NAME, "bash", "-c", bash_cmd]
            
            try:
                subprocess.run(full_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f" -> [ERRO] Falha no Bash. Código: {e.returncode}")
                continue 
        else:
            print(" -> Passo 1: Segmentações base já existem (Skipping).")

        # ---------------------------------------------------------
        # PASSO 2: REFINAMENTO SPOT (Python Puro)
        # ---------------------------------------------------------
        if os.path.exists(face_mask) and os.path.exists(eyes_raw):
            # Verificar se a máscara final já existe para não repetir trabalho
            final_mask_path = os.path.join(final_output_dir, "spot_final_combined.nii.gz")
            
            if os.path.exists(final_mask_path):
                 print(" -> Passo 2: Máscara final já existe. [PRONTO]")
                 success_count += 1
                 continue

            print(" -> Passo 2: A refinar e combinar máscaras (Spot v4)...")
            
            if not os.path.exists(final_output_dir): os.makedirs(final_output_dir)
            
            try:
                # Instanciar o Segmentador
                segmenter = SpotSegmenter(face_mask, eyes_raw, final_output_dir)
                
                # Processar (Nariz, Boca, Orelhas, Olhos)
                segmenter.process()
                
                # Guardar a máscara combinada (Labels 1-5)
                # Esta é a máscara "Ground Truth" para o teu treino
                out_path = segmenter.save_final_mask()
                
                print(f" -> [SUCESSO] Máscara de treino gerada: spot_final_combined.nii.gz")
                success_count += 1
                
            except Exception as e:
                print(f" -> [ERRO PYTHON]: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(" -> [ERRO] Ficheiros intermédios em falta.")

    print("-" * 30)
    print(f"Dataset de Treino Preparado.")
    print(f"Total Processado com Sucesso: {success_count} / {total}")

if __name__ == "__main__":
    run_training_prep()