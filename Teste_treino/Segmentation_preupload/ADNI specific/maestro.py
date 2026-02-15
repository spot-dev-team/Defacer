import os
import subprocess
import time
from tools import SpotSegmenter, SpotDefacer # Confirma se o nome do ficheiro é 'spot_tools.py' ou 'tools.py'

# --- CONFIGURAÇÃO ---
BASE_DIR = r"C:\Tese\Datasets\Defacer\ADNI_Structured"

# NOME DA DISTRO (O que usaste no comando cp: Ubuntu-22.04)
WSL_DISTRO_NAME = "Ubuntu-22.04"

# CAMINHO DO SCRIPT DENTRO DO LINUX
# O til (~) refere-se à pasta onde guardaste o ficheiro no passo anterior
SCRIPT_PATH_LINUX = "~/step1_mideface.sh"

# --- FUNÇÕES AUXILIARES ---
def to_wsl_path(win_path):
    """Converte C:\Pasta para /mnt/c/Pasta (formato Linux)"""
    abs_path = os.path.abspath(win_path)
    drive, rest = os.path.splitdrive(abs_path)
    drive_letter = drive.replace(':', '').lower()
    # Nota: Usamos aspas escapadas extra caso haja espaços nos nomes
    wsl_path = f"/mnt/{drive_letter}{rest.replace(os.sep, '/')}"
    return wsl_path

# --- PIPELINE ---
def run_pipeline():
    if not os.path.exists(BASE_DIR):
        print(f"[ERRO CRÍTICO] Pasta de dados não encontrada: {BASE_DIR}")
        return

    subjects = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    subjects.sort()
    total = len(subjects)
    
    print(f"[MAESTRO] A iniciar processamento de {total} exames...")
    print(f"[MAESTRO] Modo: Usando script interno no WSL ({WSL_DISTRO_NAME})")
    
    for i, subj in enumerate(subjects):
        subj_path = os.path.join(BASE_DIR, subj)
        print(f"\n[{i+1}/{total}] Processando: {subj}")
        
        # Encontrar o ficheiro raw
        nii_files = [f for f in os.listdir(subj_path) if f.endswith(".nii.gz") and "face_mask" not in f and "anon" not in f]
        
        if not nii_files:
            print(" -> [SKIP] Nenhum NIfTI raw encontrado.")
            continue
        
        raw_file = os.path.join(subj_path, nii_files[0])
        mid_output_dir = os.path.join(subj_path, "mid_output")
        final_output_dir = os.path.join(subj_path, "spot_final")
        
        # ---------------------------------------------------------
        # PASSO 1: MIDEFACE (Bash via WSL)
        # ---------------------------------------------------------
        face_mask = os.path.join(mid_output_dir, "face_mask.nii.gz")
        eyes_raw = os.path.join(mid_output_dir, "eyes_raw.nii.gz")
        
        if not (os.path.exists(face_mask) and os.path.exists(eyes_raw)):
            print(" -> Passo 1: A chamar MiDeFace (WSL)...")
            
            # Converter caminhos dos DADOS para WSL
            wsl_raw = to_wsl_path(raw_file)
            wsl_out = to_wsl_path(mid_output_dir)
            
            # COMANDO PURO:
            # 1. wsl -d Ubuntu-22.04 (Garante a distro certa)
            # 2. bash -c "..." (Executa o comando como string no Linux)
            # 3. Chamamos o script que JÁ LÁ ESTÁ (~/step1_mideface.sh)
            bash_cmd = f"{SCRIPT_PATH_LINUX} \"{wsl_raw}\" \"{wsl_out}\""
            
            full_cmd = ["wsl", "-d", WSL_DISTRO_NAME, "bash", "-c", bash_cmd]
            
            try:
                # check=True lança erro se falhar
                subprocess.run(full_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f" -> [ERRO] Falha no Bash. Código: {e.returncode}")
                continue 
        else:
            print(" -> Passo 1: Outputs já existem (Skipping).")

        # ---------------------------------------------------------
        # PASSO 2: SPOT SEGMENTER & DEFACER (Python Puro)
        # ---------------------------------------------------------
        if os.path.exists(face_mask) and os.path.exists(eyes_raw):
            print(" -> Passo 2: A refinar máscaras e anonimizar (Spot v4)...")
            
            if not os.path.exists(final_output_dir): os.makedirs(final_output_dir)
            
            try:
                # A. Gerar Máscara Spot Final
                segmenter = SpotSegmenter(face_mask, eyes_raw, final_output_dir)
                segmenter.process()
                combined_mask = segmenter.save_final_mask()
                
                # B. Aplicar Defacing
                defacer = SpotDefacer()
                final_anon_path = os.path.join(final_output_dir, f"SPOT_ANON_{nii_files[0]}")
                
                success = defacer.Deidentification_image_nii_SPOT(
                    where=[True, True, True, True],
                    nfti_path=raw_file,
                    spot_mask_path=combined_mask,
                    dest_path=final_anon_path,
                    verif_path=final_output_dir,
                    prefix=""
                )
                
                if success: print(" -> [SUCESSO] Exame concluído.")
                else: print(" -> [FALHA] Erro no Defacer.")
                
            except Exception as e:
                print(f" -> [ERRO CRÍTICO NO PYTHON]: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(" -> [ERRO] Passo 2 abortado: Ficheiros do Passo 1 em falta.")

if __name__ == "__main__":
    run_pipeline()