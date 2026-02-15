import os
import subprocess
import time
import shutil
import random
# Importamos a tua ferramenta de segmentação existente
from tools import SpotSegmenter

# --- CONFIGURAÇÃO ---
# Pasta onde colocaste os exames IXI estruturados (raw.nii.gz)
BASE_DIR = r"D:\Tese_BDs\Defacer\IXI_Structured"

# Limite de PACIENTES ÚNICOS para processar
MAX_EXAMS = 10

# Configurações do WSL
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

def get_unique_subjects_list(base_dir, limit):
    """
    Retorna uma lista de pastas correspondentes a pacientes únicos.
    Evita processar o mesmo paciente duas vezes.
    """
    all_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    # Dicionário para agrupar exames por ID de paciente
    # Ex: {'IXI002': ['IXI002-Guys-0828...'], 'IXI012': ['IXI012-HH...']}
    patients_dict = {}
    for folder in all_folders:
        # Assume que o ID é a primeira parte antes do hífen (ajusta se necessário)
        patient_id = folder.split('-')[0]
        if patient_id not in patients_dict:
            patients_dict[patient_id] = []
        patients_dict[patient_id].append(folder)
    
    unique_ids = list(patients_dict.keys())
    print(f" -> Encontrados {len(all_folders)} exames de {len(unique_ids)} pacientes únicos.")
    
    # Seleção Aleatória (Seed fixa para ser sempre os mesmos 150 se correres de novo)
    random.seed(42)
    
    # Se tivermos menos pacientes que o limite, usamos todos
    if len(unique_ids) > limit:
        selected_ids = random.sample(unique_ids, limit)
    else:
        selected_ids = unique_ids
        
    # Converter IDs de volta para o caminho da pasta (escolhemos o 1º exame de cada paciente)
    final_folders = []
    for pid in selected_ids:
        # Pega no primeiro exame disponível desse paciente
        folder_name = patients_dict[pid][0]
        final_folders.append(folder_name)
        
    return final_folders

def run_ixi_pipeline():
    if not os.path.exists(BASE_DIR):
        print(f"[ERRO] Pasta não encontrada: {BASE_DIR}")
        return

    print(f"--- A SELECIONAR PACIENTES ÚNICOS ---")
    subjects_to_process = get_unique_subjects_list(BASE_DIR, MAX_EXAMS)
    
    print("-" * 40)
    print(f"--- A INICIAR PIPELINE IXI (GROUND TRUTH) ---")
    print(f"Meta: Processar {len(subjects_to_process)} pacientes únicos.")
    print("-" * 40)

    success_count = 0
    error_count = 0

    for i, subj in enumerate(subjects_to_process):
        print(f"\n" + "="*60)
        print(f"[{i+1}/{len(subjects_to_process)}] A PROCESSAR: {subj}")
        print("="*60)
        
        subj_path = os.path.join(BASE_DIR, subj)
        
        # 1. Encontrar o RAW
        raw_path = os.path.join(subj_path, "raw.nii.gz")
        if not os.path.exists(raw_path):
            print(" -> [ERRO] raw.nii.gz não encontrado. Saltando.")
            error_count += 1
            continue

        # Caminhos de Output
        mid_output_dir = os.path.join(subj_path, "mid_output")
        final_output_dir = os.path.join(subj_path, "training_masks") 
        
        face_mask = os.path.join(mid_output_dir, "face_mask.nii.gz")
        eyes_raw = os.path.join(mid_output_dir, "eyes_raw.nii.gz")
        final_mask_target = os.path.join(final_output_dir, "mask_4_classes.nii.gz")

        # --- CHECKPOINT ---
        if os.path.exists(final_mask_target):
            print(" -> [INFO] Exame já processado completo. Saltando.")
            success_count += 1
            continue

        # ==============================================================================
        # PASSO 1: FREESURFER / MIDEFACE (WSL) - AGORA COM LOGS VISÍVEIS
        # ==============================================================================
        if not (os.path.exists(face_mask) and os.path.exists(eyes_raw)):
            print(" -> Passo 1: A executar MiDeFace no WSL...")
            print("    (Podes acompanhar o progresso abaixo vvv)")
            print("-" * 20)
            
            wsl_input = to_wsl_path(raw_path)
            wsl_output = to_wsl_path(mid_output_dir)
            
            # Comando WSL
            cmd = f'wsl -d {WSL_DISTRO_NAME} bash {SCRIPT_PATH_LINUX} "{wsl_input}" "{wsl_output}"'
            
            # ALTERAÇÃO AQUI: Removemos capture_output=True para ele imprimir direto no terminal
            try:
                # O Python agora espera e deixa o WSL escrever no teu ecrã
                result = subprocess.run(cmd, shell=True) 
                
                print("-" * 20)
                if result.returncode != 0:
                    print(f" -> [ERRO CRÍTICO WSL] O código de saída foi {result.returncode}.")
                    print("    Verifica a mensagem de erro acima.")
                    error_count += 1
                    continue
            except Exception as e:
                print(f" -> [ERRO EXECUÇÃO]: {e}")
                error_count += 1
                continue
        else:
            print(" -> Passo 1: outputs do MiDeFace já existem. Avançando.")

        # ==============================================================================
        # PASSO 2: REFINAMENTO PYTHON
        # ==============================================================================
        print(" -> Passo 2: A gerar Ground Truth (Separar Nariz, Boca, etc)...")
        
        if not os.path.exists(final_output_dir): os.makedirs(final_output_dir)
        
        try:
            segmenter = SpotSegmenter(face_mask, eyes_raw, final_output_dir)
            segmenter.process()
            generated_path = segmenter.save_final_mask()
            
            if os.path.exists(generated_path):
                shutil.move(generated_path, final_mask_target)
                print(f" -> [SUCESSO] Máscara Final Criada: {final_mask_target}")
                success_count += 1
            else:
                print(" -> [ERRO] O Segmentador não gerou o ficheiro final.")
                error_count += 1

        except Exception as e:
            print(f" -> [ERRO PYTHON]: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    print("="*40)
    print("PROCESSAMENTO IXI CONCLUÍDO")
    print(f"Sucessos: {success_count}")
    print(f"Erros: {error_count}")
    print("="*40)

if __name__ == "__main__":
    run_ixi_pipeline()