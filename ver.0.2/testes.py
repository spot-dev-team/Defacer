'''import numpy as np


def onehot2label(onehot_array):

    #a onehot_Array vem algo como (x, y, z, [0.1, 0.8, 0.05, 0.02, 0.03])
    onehot_array = np.argmax(onehot_array, axis=-1) #argmax pega na lista do último eixo e devolve o índice do valor máximo, assim insere qual é a label (0-4) em cada voxel
    print(f"1:\n {onehot_array} ")

    label = onehot_array[..., np.newaxis]
    print(f"2:\n {label} ")


    return label

#estrutura do exame, onde cada lista é o mapa de probabilidades de 1 voxel.
onehot_array = np.array(
    [
        [
            [  [0.1, 0.8, 0.05, 0.02, 0.03], [0.6, 0.1, 0.1, 0.1, 0.1]  ],
            [  [0.2, 0.2, 0.2, 0.2, 0.2], [0.05, 0.05, 0.05, 0.8, 0.05]  ]
        ]
    ]
                        )

onehot2label( onehot_array)'''

from defacer_version2 import Defacer as dv


# Caminhos (Ajusta para os teus ficheiros REAIS)
nifti_original = r"C:\Tese\Datasets\Defacer\ADNI_Nifti_Single_Folder\002_S_0413_MPRAGE_SENSE_20061115141346_501.nii.gz"
spot_mask = r"C:\Tese\Datasets\Defacer\ADNI_Nifti_Single_Folder\Outputs_Spot\Spot_Final_Masks\teste12\spot_final_combined.nii.gz"


# Onde guardar
output_path = r"C:\Tese\Datasets\Defacer\Testes_Spot\saida.nii.gz"
verif_path = r"C:\Tese\Datasets\Defacer\Testes_Spot\QA"


# Inicializar
defacer = dv()


# Configuração (Tudo a True para testar tudo)
where = [True, True, True, True] # Olhos, Nariz, Orelhas, Boca


# Correr a função NOVA
defacer.Deidentification_image_nii_SPOT(where, nifti_original, spot_mask, output_path, verif_path, "spot_test")