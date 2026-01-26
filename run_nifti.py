import sys
import os

# 1. Configuração de Caminhos
# Garante que o Python encontra o defacer.py
sys.path.append(os.path.join(os.getcwd(), "model_distribution"))

from defacer import Defacer

# Caminho para o ficheiro NIfTI que criaste com o dcm2niix
nifti_input_file = "" 

nifti_folder= "C:\\Tese\\Datasets\\Defacer\\ADNI_Nifti_Single_Folder"

# Pastas de saída
output_dir = "./anonymized_nifti_output"
verification_dir = "./verification_images_nii"

if not os.path.exists(output_dir): os.makedirs(output_dir)
if not os.path.exists(verification_dir): os.makedirs(verification_dir)

# 2. Inicializar o Defacer
print("Loading model...")
coder = Defacer()

# 3. Definições de Anonimização
# (Olhos, Nariz, Orelhas, Boca)
deid_mask = (1, 1, 1, 1)

# 4. Executar o Processo para NIfTI

for file in os.listdir(nifti_folder):
    if file.endswith(".nii") or file.endswith(".nii.gz"):
        nifti_input_file = os.path.join(nifti_folder, file)
        
        nifti_file_name = os.path.basename(nifti_input_file)
        if nifti_file_name.endswith(".nii.gz"):
            pass
        else:
            nifti_file_name = nifti_file_name + ".nii.gz"

        print(f"Starting de-identification for: {nifti_input_file}")

        # 'dest_path' para NIfTI deve ser o caminho completo do ficheiro de saída
        # ou a pasta onde ele vai ser guardado, dependendo da alteração no defacer.py
        result = coder.Deidentification_image_nii(
            where=deid_mask,
            nfti_path=nifti_input_file,
            dest_path=os.path.join(output_dir, f"DeID_{nifti_file_name}"),
            verif_path=verification_dir,
            url="", 
            prefix="DeID_"
        )

        if result["success"]:
            print(f"Success! NIfTI anonymized and Mask saved.")
            print(f"Check your results in: {output_dir}")
        else:
            print(f"Error: {result.get('msg')}")