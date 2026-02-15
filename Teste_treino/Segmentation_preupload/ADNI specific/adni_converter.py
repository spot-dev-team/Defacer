import os
import subprocess
import sys

# --- CONFIGURAÇÃO ---

# Onde estão as pastas originais do ADNI (Raiz que contém os Patient IDs)
INPUT_ROOT = r"D:\Tese_BDs\Defacer\ADNI"

# Onde vai ficar o dataset "limpo" para o Maestro
OUTPUT_ROOT = r"D:\Tese_BDs\Defacer\ADNI_Structured"

# Caminho para o conversor dcm2niix (ALTERA ISTO PARA O TEU CAMINHO)
DCM2NIIX_PATH = r"D:\Transferências D\dcm2niix_win\dcm2niix.exe"

# --- SCRIPT ---

def convert_adni_dataset():
    if not os.path.exists(DCM2NIIX_PATH):
        print(f"[ERRO] dcm2niix não encontrado em: {DCM2NIIX_PATH}")
        return

    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    print(f"[Conversor] A ler estrutura do ADNI em: {INPUT_ROOT}")
    
    # Listar Pacientes (002_S_0413, etc.)
    patients = [p for p in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, p))]
    print(f"[Conversor] {len(patients)} pacientes encontrados.")

    count = 0
    errors = 0

    for patient_id in patients:
        patient_path = os.path.join(INPUT_ROOT, patient_id)
        
        # Procurar pasta MPRAGE
        # Nota: Às vezes pode chamar-se "MP-RAGE" ou variações. 
        # Vamos assumir que contém "MPRAGE" no nome.
        modality_folders = [d for d in os.listdir(patient_path) if "MPRAGE" in d.upper()]
        
        if not modality_folders:
            print(f" -> [AVISO] {patient_id}: Nenhuma pasta MPRAGE encontrada. (Skipping)")
            continue

        for mod_folder in modality_folders:
            mod_path = os.path.join(patient_path, mod_folder)
            
            # Listar Datas (2008-07-31_...)
            date_folders = [d for d in os.listdir(mod_path) if os.path.isdir(os.path.join(mod_path, d))]
            
            for date_str in date_folders:
                date_path = os.path.join(mod_path, date_str)
                
                # Listar Image IDs (I115006, etc.)
                image_folders = [d for d in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, d))]
                
                for image_id in image_folders:
                    dicom_dir = os.path.join(date_path, image_id)
                    
                    # --- CONSTRUÇÃO DO OUTPUT ---
                    # Criamos um nome único: Paciente__Data__ImageID
                    # O separador duplo "__" facilita o split depois.
                    # Limpamos a data para remover caracteres chatos se necessário
                    safe_date = date_str.replace(":", "").replace(".", "")
                    unique_folder_name = f"{patient_id}__{safe_date}__{image_id}"
                    
                    target_dir = os.path.join(OUTPUT_ROOT, unique_folder_name)
                    
                    # Se a pasta já existe e tem o raw.nii.gz, saltamos
                    if os.path.exists(os.path.join(target_dir, "raw.nii.gz")):
                        print(f" -> [SKIP] Já existe: {unique_folder_name}")
                        continue
                        
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)

                    # --- COMANDO DE CONVERSÃO ---
                    # -z y : Comprimir (gzip)
                    # -f raw : Forçar o nome do ficheiro a ser 'raw.nii.gz' (Fica raw.nii.gz)
                    # -o ... : Pasta de output
                    # -i n : Ignorar ficheiros derivados (opcional, bom para o ADNI)
                    cmd = [
                        DCM2NIIX_PATH,
                        "-z", "y",
                        "-f", "raw",
                        "-o", target_dir,
                        dicom_dir
                    ]
                    
                    try:
                        # Executa o dcm2niix
                        # capture_output=True esconde o spam do dcm2niix, mostra só se der erro
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                        print(f" -> [OK] Convertido: {unique_folder_name}")
                        count += 1
                        
                        # LIMPEZA EXTRA (Opcional)
                        # O dcm2niix às vezes gera 'raw_a.nii.gz' (scout) e 'raw.nii.gz' (volume).
                        # Ou gera ficheiros JSON. Podemos querer apagar o JSON.
                        json_file = os.path.join(target_dir, "raw.json")
                        if os.path.exists(json_file):
                            os.remove(json_file)
                            
                    except subprocess.CalledProcessError as e:
                        print(f" -> [ERRO] Falha ao converter {unique_folder_name}")
                        # print(e.stderr.decode()) # Descomenta para ver o erro detalhado
                        errors += 1

    print("-" * 30)
    print(f"Processamento terminado.")
    print(f"Sucessos: {count}")
    print(f"Erros: {errors}")
    print(f"Dados prontos em: {OUTPUT_ROOT}")

if __name__ == "__main__":
    convert_adni_dataset()