import numpy as np


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

onehot2label( onehot_array)