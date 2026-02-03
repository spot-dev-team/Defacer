import os
import sys
import glob
import math
import time
import datetime as dt
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import random
import pydicom
import scipy.ndimage
from scipy import ndimage
from scipy.ndimage import shift, center_of_mass
from skimage import morphology
from skimage import measure
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt
from matplotlib import colors
from keras.utils import to_categorical
from skimage.filters import threshold_triangle
from skimage.measure import marching_cubes_lewiner
import model_ver_mouth as model


class Defacer(object):

    # onehot results -> argmax

    #da rede neuronal vem uma mapa de 5 canais (olhos, nariz, orelhas, boca, fundo) com a probabilidade de cada voxel pertencer a cada canal
    # esta função converte isso numa label única por voxel (0-4)
    #estrutura onehot_array = 
    '''X: Dimensão em X
        Y: Dimensão em Y
        Z: Dimensão em Z
        W: Estrutura com os mapas de probabilidades para todos os voxels do exame'''
    

    def Deidentification_image_nii_SPOT(self, where, nfti_path, spot_mask_path, dest_path, verif_path, prefix):
        from skimage.filters import threshold_triangle
        # ADICIONADO: shift e center_of_mass para a expansão direcional
        from scipy.ndimage import binary_dilation, gaussian_filter, generate_binary_structure, binary_closing, binary_opening, binary_fill_holes, shift, center_of_mass
        
        config = dict()
        config["resizing"] = True
        config["input_shape"] = [128, 128, 128, 1]
        prefix += "_{}"

        try:
            # 1. Carregar e Normalizar
            print(f"[Spot] A processar: {os.path.basename(nfti_path)}")
            raw_img = nib.load(nfti_path)
            raw_img = nib.as_closest_canonical(raw_img) 
            array_img = raw_img.get_fdata()
            
            spot_img = nib.load(spot_mask_path)
            spot_img = nib.as_closest_canonical(spot_img)
            spot_data = spot_img.get_fdata()

            if array_img.shape != spot_data.shape:
                raise Exception(f"Dimensões incompatíveis: {array_img.shape} vs {spot_data.shape}")

            # 2. Mapeamento
            remapped_labels = np.zeros_like(spot_data)
            remapped_labels[spot_data == 5] = 1 # Olhos
            remapped_labels[spot_data == 4] = 2 # Nariz
            remapped_labels[spot_data == 2] = 3 # Orelhas
            remapped_labels[spot_data == 3] = 4 # Boca
            
            # --- OTIMIZAÇÃO DAS MÁSCARAS ---
            print("[Spot] A preparar máscaras...")
            struct = generate_binary_structure(3, 1)

            # A. Nariz (Dilatar 1x)
            if where[1]:
                mask_nose = (remapped_labels == 2)
                if np.any(mask_nose):
                    mask_nose_dilated = binary_dilation(mask_nose, structure=struct, iterations=1)
                    remapped_labels[mask_nose_dilated] = 2 

            # B. Orelhas (Dilatar 2x)
            if where[2]:
                mask_ears = (remapped_labels == 3)
                if np.any(mask_ears):
                    mask_ears_dilated = binary_dilation(mask_ears, structure=struct, iterations=2)
                    remapped_labels[mask_ears_dilated] = 3 

            # C. Boca (Dilatar 2x)
            if where[3]:
                mask_mouth = (remapped_labels == 4)
                if np.any(mask_mouth):
                    mask_mouth_dilated = binary_dilation(mask_mouth, structure=struct, iterations=2)
                    remapped_labels[mask_mouth_dilated] = 4
            
            # D. OLHOS (EXPANSÃO DIRECIONAL SEGURA)
            if where[0]: 
                mask_eyes = (remapped_labels == 1)
                if np.any(mask_eyes):
                    # 1. Limpeza Inicial (Esfera Sólida)
                    mask_eyes = binary_opening(mask_eyes, structure=struct, iterations=1)
                    mask_eyes = binary_closing(mask_eyes, structure=struct, iterations=3)
                    mask_eyes_base = binary_fill_holes(mask_eyes)
                    
                    # --- INÍCIO DA LÓGICA DIRECIONAL ---
                    print("    -> A calcular expansão anterior (sem risco cerebral)...")
                    
                    # a. Determinar vetor de direção (Centro da Imagem -> Centro dos Olhos)
                    img_center = np.array(array_img.shape) / 2.0
                    eye_center = center_of_mass(mask_eyes_base)
                    direction_vector = eye_center - img_center
                    # Normalizar o vetor
                    direction_vector = direction_vector / np.linalg.norm(direction_vector)
                    
                    # b. Identificar o eixo dominante e a direção (para a trava de segurança)
                    dom_axis = np.argmax(np.abs(direction_vector))
                    moving_positive = direction_vector[dom_axis] > 0
                    
                    # Encontrar o limite posterior original (o ponto mais atrás que não podemos cruzar)
                    eye_indices = np.where(mask_eyes_base)
                    if moving_positive:
                        posterior_limit_idx = np.min(eye_indices[dom_axis])
                    else:
                        posterior_limit_idx = np.max(eye_indices[dom_axis])

                    # c. Shift-and-Accumulate (Translação e Acumulação)
                    # Vamos dar 10 passos para a frente, acumulando a máscara.
                    num_steps = 10 
                    step_size_voxels = 1.5 # Tamanho de cada passo
                    
                    cumulative_mask = mask_eyes_base.copy()
                    
                    for i in range(1, num_steps + 1):
                        shift_amount = direction_vector * i * step_size_voxels
                        # shift: move a máscara. order=0 mantém binário.
                        shifted_mask = shift(mask_eyes_base.astype(float), shift_amount, order=0, mode='constant', cval=0.0)
                        cumulative_mask = cumulative_mask | (shifted_mask > 0.5)
                        
                    # d. Trava de Segurança Posterior (CRUCIAL)
                    # Garante que nada foi pintado "atrás" do limite original do olho.
                    safety_mask = np.ones_like(cumulative_mask, dtype=bool)
                    slicer = [slice(None)] * 3
                    if moving_positive:
                        # Se movemos para +, zeramos tudo o que está antes do limite mínimo original
                        slicer[dom_axis] = slice(0, posterior_limit_idx)
                        safety_mask[tuple(slicer)] = False
                    else:
                        # Se movemos para -, zeramos tudo o que está depois do limite máximo original
                        slicer[dom_axis] = slice(posterior_limit_idx + 1, None)
                        safety_mask[tuple(slicer)] = False
                        
                    mask_eyes_expanded = cumulative_mask & safety_mask
                    # --- FIM DA LÓGICA DIRECIONAL ---

                    remapped_labels[mask_eyes_expanded] = 1 

            # 3. QA
            if not os.path.isdir(verif_path):
                os.makedirs(verif_path)
            fileName = os.path.basename(dest_path)
            self._generate_spot_qa_v3(array_img, remapped_labels, verif_path, fileName)

            # 4. APLICAR TRANSFORMAÇÕES (Intensidade Constante - Igual ao Original)
            print("[Spot] A aplicar algoritmos originais (Intensity Flattening)...")

            # --- NARIZ (Wipe) ---
            if where[1]:
                mask_nose = (remapped_labels == 2)
                if np.any(mask_nose):
                    array_img[mask_nose] = 0 

            # --- ORELHAS (Noise) ---
            if where[2]:
                mask_ears = (remapped_labels == 3)
                if np.any(mask_ears):
                    try:
                        thresh_air = threshold_triangle(array_img)
                    except:
                        thresh_air = np.mean(array_img) * 0.1 
                    noise = np.random.rand(*array_img.shape) * thresh_air * 0.8
                    array_img[mask_ears] = noise[mask_ears]

            # --- OLHOS (Intensity Flattening na Máscara Expandida) ---
            # D. OLHOS (DILATAÇÃO BASE + EXPANSÃO DIRECIONAL)
            if where[0]: 
                mask_eyes = (remapped_labels == 1)
                if np.any(mask_eyes):
                    # 1. Limpeza Inicial
                    mask_eyes = binary_opening(mask_eyes, structure=struct, iterations=1)
                    mask_eyes = binary_closing(mask_eyes, structure=struct, iterations=3)
                    mask_raw = binary_fill_holes(mask_eyes)
                    
                    # --- CÁLCULO DA DIREÇÃO E LIMITES ---
                    print("    -> A calcular expansão híbrida (Dilatação + Shift)...")
                    
                    # a. Vetor de Direção (Centro Cabeça -> Olhos)
                    img_center = np.array(array_img.shape) / 2.0
                    eye_center = center_of_mass(mask_raw)
                    direction_vector = eye_center - img_center
                    direction_vector = direction_vector / np.linalg.norm(direction_vector)
                    
                    # b. Identificar limite posterior ORIGINAL (Segurança)
                    # Queremos dilatar, mas não queremos que cresça para trás.
                    dom_axis = np.argmax(np.abs(direction_vector))
                    moving_positive = direction_vector[dom_axis] > 0
                    
                    eye_indices = np.where(mask_raw)
                    if moving_positive:
                        # Se o olho está a ir para +X, o limite traseiro é o Mínimo X original
                        posterior_limit_idx = np.min(eye_indices[dom_axis])
                    else:
                        # Se o olho está a ir para -X, o limite traseiro é o Máximo X original
                        posterior_limit_idx = np.max(eye_indices[dom_axis])

                    # --- PASSO 1: DILATAÇÃO ISOTRÓPICA (O teu pedido) ---
                    # Expandimos para todos os lados para apanhar o olho todo
                    mask_base = binary_dilation(mask_raw, structure=struct, iterations=2)

                    # --- PASSO 2: EXPANSÃO DIRECIONAL (SHIFT) ---
                    # Projetamos essa base alargada para a frente
                    cumulative_mask = mask_base.copy()
                    num_steps = 10 
                    step_size_voxels = 1.5
                    
                    for i in range(1, num_steps + 1):
                        shift_amount = direction_vector * i * step_size_voxels
                        shifted_mask = shift(mask_base.astype(float), shift_amount, order=0, mode='constant', cval=0.0)
                        cumulative_mask = cumulative_mask | (shifted_mask > 0.5)
                        
                    # --- PASSO 3: CORTE DE SEGURANÇA ---
                    # Cortamos tudo o que, devido à dilatação do Passo 1, 
                    # tenha crescido para trás do limite original.
                    safety_mask = np.ones_like(cumulative_mask, dtype=bool)
                    slicer = [slice(None)] * 3
                    
                    if moving_positive:
                        # Cortar tudo antes do inicio original do olho
                        slicer[dom_axis] = slice(0, posterior_limit_idx)
                        safety_mask[tuple(slicer)] = False
                    else:
                        # Cortar tudo depois do fim original do olho
                        slicer[dom_axis] = slice(posterior_limit_idx + 1, None)
                        safety_mask[tuple(slicer)] = False
                        
                    mask_eyes_final = cumulative_mask & safety_mask
                    
                    # Atualizar labels para o QA
                    remapped_labels[mask_eyes_final] = 1 

                    # --- APLICAÇÃO DA OFUSCAÇÃO (FLATTENING) ---
                    # Usamos a mesma lógica de "Intensity Flattening"
                    blurred_img = gaussian_filter(array_img, sigma=3)
                    
                    # Amostrar pele apenas na parte frontal da máscara
                    front_slicer = [slice(None)] * 3
                    if moving_positive: front_slicer[dom_axis] = slice(posterior_limit_idx + 5, None)
                    else: front_slicer[dom_axis] = slice(0, posterior_limit_idx - 5)
                    
                    mask_front = mask_eyes_final.copy()
                    mask_front[tuple(front_slicer)] = False
                    mask_front = mask_eyes_final & ~mask_front # Apenas a "capa" da frente
                    
                    if np.any(mask_front):
                         target_value = np.max(blurred_img[mask_front])
                    else:
                         target_value = np.max(blurred_img[mask_eyes_final])

                    array_img[mask_eyes_final] = target_value
                    print(f" -> Olhos: Expandidos e Achatados (Valor: {int(target_value)})")

            # --- BOCA (Intensity Flattening) ---
            if where[3]:
                mask_mouth = (remapped_labels == 4)
                if np.any(mask_mouth):
                    blurred_img = gaussian_filter(array_img, sigma=3)
                    target_value = np.max(blurred_img[mask_mouth])
                    array_img[mask_mouth] = target_value
                    print(f" -> Boca: Achatamento de intensidade (Valor: {int(target_value)})")

            # 5. Guardar
            final_img = nib.Nifti1Image(array_img, raw_img.affine, raw_img.header)
            save_p = os.path.join(os.path.dirname(dest_path), prefix.format(os.path.basename(nfti_path)))
            nib.save(final_img, save_p)
            
            return {"success": True, "msg": "Spot Processing Complete"}
            
        except Exception as ex:
            print(f"ERRO: {ex}")
            import traceback
            traceback.print_exc()
            return {"success": False, "msg": str(ex)}

    # ---------------------------------------------------------
    # QA UPGRADE: RELATÓRIO MULTI-VIEW (Axial + Sagital + Coronal)
    # ---------------------------------------------------------
    def _generate_spot_qa_v3(self, image, labels, save_path, file_name):
        import matplotlib.pyplot as plt
        from matplotlib import colors
        
        # Mapa de cores: 1=Red, 2=Purple, 3=Blue, 4=Yellow
        cmap = colors.ListedColormap(['red', 'purple', 'blue', 'yellow'])
        bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        # Configurar a figura: Agora é mais alta para caberem as vistas extra
        # 4 colunas (Órgãos) x 3 linhas (Vistas: Axial, Sagital, Coronal)
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        plt.suptitle(f"Relatório Spot 3-Axis: {file_name}", fontsize=16)

        # Helper: Encontra o slice central com base na máscara
        def get_best_slice(label_id, axis):
            # Soma pixels nos outros dois eixos para encontrar o pico
            # axis=0 (Sagital), axis=1 (Coronal), axis=2 (Axial)
            axes_to_sum = tuple(i for i in range(3) if i != axis)
            mask_counts = np.sum(labels == label_id, axis=axes_to_sum)
            if np.sum(mask_counts) == 0: 
                return int(image.shape[axis]/2)
            return np.argmax(mask_counts)

        # Definição dos Órgãos e Colunas
        organs = [
            (1, "Olhos"),   # Coluna 0
            (2, "Nariz"),   # Coluna 1
            (3, "Orelhas"), # Coluna 2
            (4, "Boca")     # Coluna 3
        ]
        
        # Loop pelos órgãos (Colunas)
        for col_idx, (lbl_id, name) in enumerate(organs):
            
            # --- LINHA 1: AXIAL (Visto de Cima - Eixo Z) ---
            ax_axial = axes[0, col_idx]
            z_slice = get_best_slice(lbl_id, axis=2)
            
            # Rotação para ficar "em pé"
            bg_ax = np.rot90(image[:, :, z_slice])
            mask_ax = np.rot90(labels[:, :, z_slice])
            
            ax_axial.imshow(bg_ax, cmap='gray')
            ax_axial.imshow(np.ma.masked_where(mask_ax != lbl_id, mask_ax), cmap=cmap, norm=norm, alpha=0.7)
            ax_axial.set_title(f"{name} - Axial (Z={z_slice})")
            ax_axial.axis('off')

            # --- LINHA 2: SAGITAL (Visto de Lado - Eixo X) ---
            ax_sag = axes[1, col_idx]
            x_slice = get_best_slice(lbl_id, axis=0)
            
            # Corte Sagital: Y vs Z. Rodamos para o Z ficar para cima.
            bg_sag = np.rot90(image[x_slice, :, :])
            mask_sag = np.rot90(labels[x_slice, :, :])
            
            ax_sag.imshow(bg_sag, cmap='gray')
            ax_sag.imshow(np.ma.masked_where(mask_sag != lbl_id, mask_sag), cmap=cmap, norm=norm, alpha=0.7)
            ax_sag.set_title(f"{name} - Sagital (X={x_slice})")
            ax_sag.axis('off')

            # --- LINHA 3: CORONAL (Visto de Frente - Eixo Y) ---
            ax_cor = axes[2, col_idx]
            y_slice = get_best_slice(lbl_id, axis=1)
            
            # Corte Coronal: X vs Z. Rodamos para o Z ficar para cima.
            bg_cor = np.rot90(image[:, y_slice, :])
            mask_cor = np.rot90(labels[:, y_slice, :])
            
            ax_cor.imshow(bg_cor, cmap='gray')
            ax_cor.imshow(np.ma.masked_where(mask_cor != lbl_id, mask_cor), cmap=cmap, norm=norm, alpha=0.7)
            ax_cor.set_title(f"{name} - Coronal (Y={y_slice})")
            ax_cor.axis('off')

        save_file = os.path.join(save_path, f'QA_MultiView_{file_name}.png')
        plt.tight_layout()
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()
        print(f"[Spot QA] Relatório Multi-View salvo em: {save_file}")





    def onehot2label(self, onehot_array):

        #a onehot_Array vem algo como (x, y, z, [0.1, 0.8, 0.05, 0.02, 0.03])
        onehot_array = np.argmax(onehot_array, axis=-1) #argmax pega na lista do último eixo e devolve o índice do valor máximo, assim insere qual é a label (0-4) em cada voxel
        exam_labeled = onehot_array[..., np.newaxis] #argmax devolve (x,y,z), este newaxis insere o label num eixo novo no final, ficando (x,y,z,label)

        return exam_labeled


    # Make the superior coordinates the direction of increasing.
    #mete o exame de pé
    def flip_axis(self, x, axis):
        x = np.asarray(x).swapaxes(axis, 0) #coloca o eixo que queremos corrigir na primeira posição para ser mais fácil de o processar a nível de código
        x = x[::-1, ...] #inverte o eixo que está na primeira posição
        x = x.swapaxes(0, axis) #devolve o eixo à sua posição original
        return x
    

    # Loop over the image files and store everything into a list.
    def load_scan(self, list_test_image):
        slices = [pydicom.read_file(s)
                  for s in list_test_image if s.endswith(".dcm")] #lê todos os ficheiros DICOM da pasta e guarda-os numa lista
                                                                  # usa o pydicom para ler os ficheiros DICOM
        slices.sort(key=lambda x: int(x.InstanceNumber))  # ordena a lista de slices pelo InstanceNumber (número do slice no exame)

        return slices

    # Merge dicom image 2D to 3D
    def get_pixels(self, scans):
        # default stack axis = 0 // pixel_array function import [y, x], 3D array becomes [z y x]
        image = np.stack([s.pixel_array for s in scans]) #Pega nos dados dos pixels de cada slice e empilha-os num array 3D

        # Convert to int16 (from sometimes int16),
        # values should always be low enough (<32k)
        # image = image.astype(np.int16)

        return image


    # Delete dicom's header info - metadata
    def header_deidentification(self, scans, check=True):
        de_code_list = [0x00080012,  # Instance Creation Date
                        0x00080013,  # Instance Creation Time
                        0x00080020,  # Study Date
                        0x00080021,  # Series Date
                        0x00080022,  # Acquisition Date
                        0x00080023,  # Image Date, Content Date
                        0x00080030,  # Study Time
                        0x00080031,  # Series Time
                        0x00080032,  # Acquisition Time
                        0x00080033,  # Image Time, Content Time
                        0x00080050,  # Accession Number
                        0x00080080,  # Institution name
                        0x00080081,  # Institution Address
                        0x00080090,  # Referring Physician's name
                        0x00081010,  # Station name
                        0x00081040,  # Institutional Department name
                        0x00081070,  # Operator's Name
                        0x00100010,  # Patient's name
                        0x00100020,  # Patient's ID
                        0x00100030,  # Patient's Birth Date
                        0x00100040,  # Patient's Sex
                        0x00101010,  # Patient's Age
                        0x00204000]  # Image Comments

        for s in scans:
            for code in de_code_list:
                # If present, replace with spaces
                try:
                    s[code].value = ''
                except:
                    pass


        if check == True:  # check option - apenas um sanity check para garantir que os dados foram apagados
            # s[0x00200013].value: Instance Number
            print('dicom Instance Number:', scans[0][0x00200013].value, '\n')
            for code in de_code_list:
                try:
                    print('DE-IDENTIFIED : ', s[code])
                except:
                    pass

    # Find eyes and nods
    def bounding_box(self, results):
        boxes = list() #vai armazenar as coordenadas de cada caixa delimitadora
        
        #results formato: $$(Batch, Z, Y, X, Classes)$$Exemplo real com valores:$$(1, 128, 128, 128, 5)$$

        for ch in range(results.shape[-1]):  # .shape[-1] acede ao número de classes
    
            #except 0 label (blanck)
            if ch == 0 or ch == 2:  # eyes, ears (pois há 2 de cada um, então é lógico)

                result = np.round(results[..., ch]) #pega no mapa de probabilidades do canal atual e arredonda os valores (0.8 -> 1, 0.2 -> 0)
                lb = label(result, connectivity=1) #agrupa os voxels ligados com valores correspondentes a regiões rotuladas e dá-lhes um número de label
                # lb= Voxels da Ilha A = 1 
                # Voxels da Ilha B = 2
                # Voxels da Ilha C = 3
                # Voxels da Ilha D = 4

                if np.max(lb) > 2: #se houver mais de 2 regiões rotuladas (mais de 2 olhos ou 2 orelhas) faz isto para controlar o ruído
                    region_list = [region.area for region in regionprops(lb)]     #cria uma lista com as áreas de cada região rotulada ex: [1000, 850, 50, 30]
                    lb = remove_small_objects(lb, min_size=np.max(region_list)*0.3) #verifica o grupo com maior área e diz que qualquer grupo com menos de 1/3 do tamanho é para ser desconsiderado pois deve ser ruído.

                if len(regionprops(lb))!=2 : #se não conseguir encontrar exatamente 2 regiões rotuladas (2 olhos ou 2 orelhas) lança um erro
                    raise Exception('Could not find proper eyes on the face')

                for region in regionprops(lb):
                    boxes.append(list(region.bbox)) #adiciona as coordenadas da caixa delimitadora de cada região rotulada à lista boxes
                    #region.bbox: Extrai os limites da ilha: (z_min, y_min, x_min, z_max, y_max, x_max)

            if ch == 1 or ch ==3 : # nose, mouth

                result = np.round(results[..., ch])
                lb = label(result, connectivity=1)

                if np.max(lb) > 1:
                    region_list = [region.area for region in regionprops(lb)]       
                    lb = remove_small_objects(lb, min_size=np.max(region_list)*0.3)

                if len(regionprops(lb))!=1 :
                    raise Exception('Could not find proper nose on the face')

                for region in regionprops(lb):
                    boxes.append(list(region.bbox))

        return boxes
    
    #nota 
    '''A ordem padrão no espaço 3D é: (z_min, y_min, x_min, z_max, y_max, x_max)
    Índices 0, 1, 2: Coordenadas Mínimas $(\min_z, \min_y, \min_x)
    Índices 3, 4, 5: Coordenadas Máximas $(\max_z, \max_y, \max_x)'''


    def dicom_view_label (self, image, labels, boxes, axial_plane, save_path, file_name):
        boxes =np.array(boxes)
        centers = (boxes[:,0:3]+boxes[:,3:6])/2 #centers of nose, right ear, left ear (verifica o ponto médio de cada bounding box)
        ''' Bounding Box do Olho Esquerdo é:[10, 20, 30, 20, 40, 50]
        Z_centro: (10 + 20) / 2 = 15
        Y_centro: (20 + 40) / 2 = 30
        X_centro: (30 + 50) / 2 = 40
        '''
        pred = np.argmax(labels,axis=-1) #pega no índice do canal com maior probabilidade para cada voxel, ficando com um array 3D onde cada voxel tem um valor entre 0-4 conforme a label atribuída        
        ones = np.ones(image.shape) #cria um array de uns com o mesmo shape que o exame original com tudo a 1. Ou seja, uma matriz de 1s
        for i in range(len(boxes)):
            ones = self.box_blur(ones, boxes[i], 0) #chama a função box_blur para cada bounding box, que vai colocar a 0 os voxels dentro da bounding box na matriz de uns
        
        ones = 1-ones #inverte a matriz de uns, ficando com 1s nas bounding boxes e 0s no resto do exame. Agora apenas as bounding boxes têm informação, o resto é "apagado" por estar a 0

        pred = pred*ones #multiplica o exame rotulado pela matriz de bounding boxes, pelas labels de cada voxel, ficando apenas com as labels dentro das bounding boxes, o resto do exame fica a 0
        
        cmap = colors.ListedColormap(['None', 'red', 'purple', 'blue', 'yellow', 'green']) #rigt eye 1, left eye 2, nose 3, right ear 4, left ear5
        bounds=[0,1,2,3,4,5,6]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        if axial_plane == 0:

            plt.figure(figsize=(15,10))
            plt.subplot(2,2,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,4)
            slice_num = int(centers[5][axial_plane])
            plt.title('predicted mouth: axial = {}'.format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

        elif axial_plane == 1:

            plt.figure(figsize=(15,10))

            plt.subplot(2,2,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,4)
            slice_num = int((centers[5][axial_plane]))
            plt.title('predicted mouth: axial = {}' .format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)


        elif axial_plane == 2:

            plt.figure(figsize=(15,10))

            plt.subplot(2,2,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,4)
            slice_num = int(centers[5][axial_plane])
            plt.title('predicted mouth: axial = {}'.format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)
    
        pic_name = os.path.join(save_path,'label_{}.png'.format(os.path.basename(file_name)))
        plt.savefig(pic_name, bbox_inches='tight')
        plt.close('all')



    # take a pic for users to check areas that this tool has found
    def nifti_view_label (self, image,labels,boxes,path,file_name): 
        pred = np.argmax(labels,axis=-1)
        
        boxes =np.array(boxes)
        centers = (boxes[:,0:3]+boxes[:,3:6])/2 #centers of nose, right ear, left ear
        axial_plane = np.argmin(np.var(centers[2:5],axis=0)) # 코와 귀 2개 좌표만 뽑아서 가장 분산이 작은 축을 구함 = axial 축일것이라 예상됨.
        
        ones = np.ones(image.shape)
        for i in range(len(boxes)):
            ones = self.box_blur(ones,boxes[i], 1)
        
        ones = 1-ones
        pred = pred*ones.T
        
        image =image.T
        
        cmap = colors.ListedColormap(['None', 'red', 'purple', 'blue', 'yellow', 'green']) #rigt eye 1, left eye 2, nose 3, right ear 4, left ear5
        bounds=[0,1,2,3,4,5,6]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        if axial_plane == 0:

            plt.figure(figsize=(15,10))

            plt.subplot(2,2,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,4)
            slice_num = int(centers[5][axial_plane])
            plt.title('predicted mouth: axial = {}'.format(slice_num))
            plt.imshow(image[slice_num,:,:],cmap='gray')
            plt.imshow(pred[slice_num,:,:],alpha=0.5,cmap=cmap, norm=norm)

        elif axial_plane == 1:

            plt.figure(figsize=(15,10))

            plt.subplot(2,2,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,4)
            slice_num = int((centers[5][axial_plane]))
            plt.title('predicted mouth: axial = {}' .format(slice_num))
            plt.imshow(image[:,slice_num,:],cmap='gray')
            plt.imshow(pred[:,slice_num,:],alpha=0.5,cmap=cmap, norm=norm)
	        
        elif axial_plane == 2:

            plt.figure(figsize=(15,10))

            plt.subplot(2,2,1)
            slice_num = int((centers[0][axial_plane]+centers[1][axial_plane])/2)
            plt.title('predicted eyes: axial = {}'.format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,2)
            slice_num = int(centers[2][axial_plane])
            plt.title('predicted nose: axial = {}'.format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,3)
            slice_num = int((centers[3][axial_plane]+centers[4][axial_plane])/2)
            plt.title('predicted ears: axial = {}' .format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)

            plt.subplot(2,2,4)
            slice_num = int(centers[5][axial_plane])
            plt.title('predicted mouth: axial = {}'.format(slice_num))
            plt.imshow(image[:,:,slice_num],cmap='gray')
            plt.imshow(pred[:,:,slice_num],alpha=0.5,cmap=cmap, norm=norm)
    
        pic_name = os.path.join(path,'label_{}.png'.format(os.path.basename(file_name)))
        plt.savefig(pic_name, bbox_inches='tight')
        plt.close('all')

        

    # wipe nose
    def box_blur(self, im_array, box, option, wth=1):
        # DICOM (option == 0) -> Ordem esperada: x, y, z
        # NIfTI (option != 0) -> Ordem esperada: z, y, x

        # increase or decrease the size of the box by 'wth' times
        if wth != 1:
            for c in range(3): #para percorrer as 3 coordenadas x,y,z ou z,y,x conforme o formato do exame
                mean_ = (box[c]+box[c+3])/2 #faz a posição média entre o mínimo e o máximo da bounding box na coordenada atual
                box[c] = int(np.round(mean_-wth*(mean_-box[c]))) #calcula o novo mínimo
                box[c+3] = int(np.round(wth*(box[c+3]-mean_)+mean_)) #calcula o novo máximo
                if box[c] < 0:
                    box[c] = 0
                if box[c+3] > im_array.shape[2-c]:
                    # order : im_array-> x,y,z / box-> z,y,x
                    box[c+3] = im_array.shape[2-c]

        # voxel coordinates must be 'int'
        # if it is dicom files
        if option == 0: 
            box_x1 = box[0]
            box_y1 = box[1]
            box_z1 = box[2]
            box_x2 = box[3]
            box_y2 = box[4]
            box_z2 = box[5]
        # or it is nfty files
        else:
            box_z1 = box[0]
            box_y1 = box[1]
            box_x1 = box[2]
            box_z2 = box[3]
            box_y2 = box[4]
            box_x2 = box[5]

        # wipe nose
        blurr_array = 0
        im_array[box_x1:box_x2, box_y1:box_y2, box_z1:box_z2] = blurr_array #substitui os valores dos voxels dentro da bounding box por 0

        return im_array

    # wipe eyes
    def surface_blur(self, im_array, edge_img, box, wth, dep, option):
        # increase or decrease the size of the box by 'wth' times
        if wth != 1:
            for c in range(3):
                mean_ = (box[c]+box[c+3])/2
                box[c] = int(np.round(mean_-wth*(mean_-box[c])))
                box[c+3] = int(np.round(wth*(box[c+3]-mean_)+mean_))
                if box[c] < 0:
                    box[c] = 0
                if box[c+3] > im_array.shape[2-c]:
                    # order : im_array-> x,y,z / box-> z,y,x
                    box[c+3] = im_array.shape[2-c]

        # voxel coordinates must be 'int'
        # if it is dicom files
        if option == 0:
            box_x1 = box[0]
            box_y1 = box[1]
            box_z1 = box[2]
            box_x2 = box[3]
            box_y2 = box[4]
            box_z2 = box[5]

            #1. Isolamento da ROI (mini_array e mini_edge)O código cria sub-volumes (crops) para poupar memória e tempo de processamento. 
            # Trabalhar num cubo de $30^3$ é muito mais rápido do que no volume total de $128^3$.
            # NOTA: Talvez não seja preciso fazer isto com HPC
            mini_array = im_array[box_x1:box_x2, box_y1:box_y2, box_z1:box_z2] #mapa de intensidades do sub-volume
            mini_edge = edge_img[box_x1:box_x2, box_y1:box_y2, box_z1:box_z2] #mapa de bordas do sub-volume
            processing_area = np.zeros_like(mini_array) #Cria um array de zeros com o formato exato da mini_array. Este array vai ser usado para marcar as áreas que precisam de ser processadas (desfocadas).

            # blur eye
            where_true = np.where(mini_edge == True)  #retorna as coordenadas dos voxels que são bordas (True) no mini_edge
            '''
            ([z1,z2,z3],[y1,y2,y3],[x1,x2,x3]) = where_true
            # Se o olho tiver uma superfície de 5.000 voxels:
            # len(where_true[0]) = 5000
            # where_true[0] = [z1, z2, z3, ..., z5000]
            # where_true[1] = [y1, y2, y3, ..., y5000]
            # where_true[2] = [x1, x2, x3, ..., x5000]
            '''

            for i in range(len(where_true[0])):
                x = where_true[0][i]
                y = where_true[1][i]
                z = where_true[2][i]
                # dep = depth (ex: dep = 2)
                processing_area[x-dep:x+dep,y-dep:y+dep,z-dep:z+dep] = 1 #expande-se a área de processamento em torno de cada voxel de borda para uma anonimização maior
            
            '''# three steps to get the threshold value
            # 1. Seleciona apenas os valores da imagem original que estão dentro da casca (ones)
            pixels_da_casca = mini_array[processing_area == 1]

            # 2. Aplica um desfoque gaussiano nesses valores
            pixels_borrados = ndimage.gaussian_filter(pixels_da_casca, sigma=3)

            # 3. Escolhe o valor mais alto desse borrão para ser a "cor" da nova superfície
            threshold = np.max(pixels_borrados)'''

            #ORIGINAL
            threshold = np.max(ndimage.gaussian_filter(mini_array[processing_area==1],sigma=3))

            #TESTE
            #threshold = np.median(ndimage.gaussian_filter(mini_array[processing_area==1],sigma=3))

            mini_array[processing_area==1] = threshold #substitui os valores dos voxels na área de processamento pelo valor do threshold obtido


        # or it is nifti files
        else:
            box_z1 = box[0]
            box_y1 = box[1]
            box_x1 = box[2]
            box_z2 = box[3]
            box_y2 = box[4]
            box_x2 = box[5]

            mini_array = im_array[box_x1:box_x2, box_y1:box_y2, box_z1:box_z2]
            mini_edge = edge_img[box_x1:box_x2, box_y1:box_y2, box_z1:box_z2]
            processing_area = np.zeros_like(mini_array)

            # blur eye
            where_true = np.where(mini_edge == True)

            for i in range(len(where_true[0])):
                x = where_true[0][i]
                y = where_true[1][i]
                z = where_true[2][i]
                processing_area[x-dep:x+dep, y-dep:y+dep, z-dep:z+dep] = 1

            threshold = np.max(ndimage.gaussian_filter(mini_array[processing_area == 1], sigma=3))

            mini_array[processing_area == 1] = threshold

        # im_array: O volume total (o cérebro inteiro)
        # mini_array: O recorte que acabaste de borrar (o "penso" cirúrgico)
        #aquie basta substituir a parte do volume total pelo recorte borrado pois não foi feito downsampling, apenas se trabalhou nos voxels de interesse
        im_array[box_x1:box_x2, box_y1:box_y2, box_z1:box_z2] = mini_array
        return im_array

    # convert image 2D to 3D shape
    def outer_contour_3D(self, image, zoom=1):
        
        # sort in standard size - calcula quanto é preciso redimensionar o exame para ficar com 128x128x128 voxels
        resize_factor = (128/image.shape[0],
                         128/image.shape[1], 
                         128/image.shape[2])
        
        # resize image para 128x128x128
        ima = ndimage.zoom(image, resize_factor, order=0, #ordem 0 = nearest neighbor isto signifca que não há interpolação, mantém os valores originais
                           mode='constant', cval=0.0)  #Preenche qualquer espaço vazio criado com zeros (preto)

        # make binary cast
        thresh = threshold_triangle(ima) #algoritmo auto para encontrar o melhor threshold para segmentar a imagem em objeto e background
        imageg = ndimage.median_filter(ima, size=3) #remove ruido para não haver pontos com valores fora do esperado
        binary_image = imageg > thresh #Cria uma máscara booleana. Tudo o que for mais brilhante que o thresh vira True (cabeça), o resto vira False (ar).

        #precorrer os 3 eixos à procura de "buracos" e se encontrar (como nas covas nasais), preenche os mesmos para criar uma só estrutura sólida
        for s in range(ima.shape[0]):
            binary_image[s, :, :] = ndimage.morphology.binary_fill_holes(
                binary_image[s, :, :])
        for s in range(ima.shape[1]):
            binary_image[:, s, :] = ndimage.morphology.binary_fill_holes(
                binary_image[:, s, :])
        for s in range(ima.shape[2]):
            binary_image[:, :, s] = ndimage.morphology.binary_fill_holes(
                binary_image[:, :, s])

        # draw outer contour - cria a casca do objeto sólido
        verts, faces, norm, val = marching_cubes_lewiner(binary_image, 0) #função auto que encontra a isosuperfície (a fronteira entre o objeto e o ar)
        vint = np.round(verts).astype('int') #Arredonda as coordenadas da máscara para inteiros, para que correspondam a voxels na grelha da imagem.
        contour = np.zeros_like(binary_image) #"Pinta" apenas esses pontos na matriz contour. O resultado é uma imagem preta com uma linha branca fina a desenhar a forma da cabeça.
        for s in vint:
            contour[s[0], s[1], s[2]] = 1

        # shrink contour image because of the gaussian_filter we used earlier.
        if zoom != 1:
            c_shape = contour.shape
            zoom_ = ndimage.zoom(contour, zoom, order=0,    #encolhe a imagem da casca para reduzir o efeito do desfoque gaussiano que será aplicado posteriormente
                                 mode='constant', cval=0.0)
            zoom_shape = zoom_.shape
            npad = ((int(np.ceil((c_shape[0]-zoom_shape[0])/2)), int((c_shape[0]-zoom_shape[0])/2)), #calcula o padding necessário para voltar ao tamanho original após o zoom
                    (int(np.ceil((c_shape[1]-zoom_shape[1])/2)),
                     int((c_shape[1]-zoom_shape[1])/2)),
                    (int(np.ceil((c_shape[2]-zoom_shape[2])/2)), int((c_shape[2]-zoom_shape[2])/2)))

            contour_3D = np.pad(zoom_, npad, 'constant', constant_values=(0)) #adiciona o padding calculado, preenchido com zeros (preto)


        elif zoom == 1:
            contour_3D = contour

        # Revert to original size
        get_back = (image.shape[0]/128, image.shape[1]/128, image.shape[2]/128)  #calcula o fator de zoom necessário para voltar ao tamanho original do exame
        contour_3D = ndimage.zoom(
            contour_3D, get_back, order=0, mode='constant', cval=0.0) #adapta a máscara ao tamanho original da imagem 

        return contour_3D

    # where_do_you_want_to_blur? ex) where = (1,1,1) -> blur(eyes, nose, ears)
    def Deidentification_image_dcm(self, where, dicom_path, dest_path, verif_path, prefix, Model=model):
        '''
        where : list or tuple. Each position stands for eyes nose ears (eyes, nose, ears) 
                If the corresponding position is 1, de-identification process.

        dicom_path : Test set(labled or unlabled) data path. 
        model : Predictive model to be applied.
        '''
        try:
            config = dict()
            config["resizing"] = True
            config["input_shape"] = (128, 128, 128, 1)

            prefix += "_{}"

            # carrega os ficheiros DICOM
            list_test_image = glob.glob(dicom_path + '/*.dcm') 
            slices = self.load_scan(list_test_image) # Lê o volume
            array_img = self.get_pixels(slices) # Extrai os valores dos voxels para um array 3D numpy            
            original_shape = array_img.shape
            d_type = array_img.dtype
            thresh = threshold_triangle(array_img)

            # Header de identification
            self.header_deidentification(slices, check=False)

            # Make the superior coordinates the direction of increasing.
            # Álgebra que permite determinar a orientação espacial do exame DICOM
            X = slices[0][0x00200037].value[0:3] #vetor X
            X = [float(i) for i in X]

            Y = slices[0][0x00200037].value[3:6] #vetor Y
            Y = [float(i) for i in Y]

            superior = [X[2], #componente Z das linhas, vista coronal
                        Y[2], #componente Z das colunas, vista sagital
                        np.cross(X,Y)[2]] #componente Z em que é a direção de "emplihamento" dos exames DICOM, vista axial
            arg = np.argmax(np.abs(superior)) #Descobre qual é o eixo dominante

            if superior[arg] < 0: #se for negativo, inverte o eixo pois a imagem está ao contrário
                image = self.flip_axis(array_img, (2 - arg))
            else:
                image = array_img

            # load prediction label
            image = model.resize(image)
            image = image.reshape(1, 128, 128, 128, 1)
            results = model.model.predict(image)

            if superior[arg] < 0:
                # +1 : index 0 is batch size
                results = self.flip_axis(results, (3 - arg))

            # preprocessing: Size recovery and transform onehot to labels number
            if config["resizing"] == True:
                results = self.onehot2label(results)
                # prediction results (batch size, dep, col ,row, ch) -> (dep, col ,row)
                results = np.reshape(results, config["input_shape"][0:3])
                results = model.resize(results,
                                       img_dep=original_shape[0],
                                       img_cols=original_shape[1],
                                       img_rows=original_shape[2])
                results = to_categorical(results)

            else:
                results = results[0, ...]  # Only if batch size==1

            # search center by clustering
            boxes = self.bounding_box(results[..., 1:])

            # view label with .png
            if not os.path.isdir(verif_path):
                os.makedirs(verif_path)
            fileName = os.path.basename(dest_path)
            self.dicom_view_label(array_img, results, boxes, (2-arg), verif_path, fileName)

            # blur parts of face
            if where[1]:  # nose
                box = boxes[2]
                array_img = self.box_blur(array_img, box, 0, wth=1.33)

            # make outer contour for mini array.
            edge_img = self.outer_contour_3D(array_img, zoom=1)

            if where[0]:  # eyes

                box = boxes[0]  # eye
                array_img = self.surface_blur(
                    array_img, edge_img, box, wth=1.5, dep=3, option=0)

                box = boxes[1]  # eye
                array_img = self.surface_blur(
                    array_img, edge_img, box, wth=1.5, dep=3, option=0)

            if where[2]:  # ears
                '''
                In order not to see the outline of the ear due to external noise,
                fill the area of the ear with similar noise
                '''
                ear_results = results[...,3]
                border = self.box_blur(np.ones(array_img.shape), boxes[3], 0) #'box_blur' function is based on array_img.shape (nibabel liabrary)
                border = self.box_blur(border, boxes[4], 0)
                border = 1-border
                ear_results = border*ear_results

                noise = np.random.rand(*original_shape)*thresh*0.8 
                array_img[ear_results == 1] = noise[ear_results == 1] 

            if where[3] : # mouth

                mouth_results = results[...,4] 
                border = self.box_blur(np.ones(array_img.shape),boxes[5],0) #'box_blur' function is based on array_img.shape (nibabel liabrary)
                border = 1-border
                if where[1] == False: # If you want to preserve the nose
                     border = self.box_blur(border,boxes[2],0, wth=1.5)

                mouth_results = border*mouth_results

                threshold = np.max(ndimage.gaussian_filter(array_img[mouth_results==1],sigma=3))
                array_img[mouth_results==1] = threshold

            array_img = np.round(array_img)
            array_img = np.array(array_img, dtype=d_type)
            # processed 3D image array

            for i in range(len(slices)):
                # [i, :, : ] the reason is that function np.stack makes new axis as first axis
                slices[i].PixelData = array_img[i, :, :].tostring()

            for i in range(len(slices)):
                instanceNum = pydicom.read_file(
                    list_test_image[i]).InstanceNumber
                # instance Number starts from 1.
                slices[instanceNum-1].save_as(os.path.join(
                    dest_path, prefix.format(os.path.basename(list_test_image[i]))))

            return {"success": True, "msg": ""}
        except Exception as ex:
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(ex).__name__, ex)
            return {"success": False, "msg": str(ex)}

    # 5D tensor (batch, img_dep, img_cols, img_rows, img_channel)
    def load_batch(self, x_list, y_list=0, batch_size=1):
        # prepara dados para o modelo

        config = dict()  # configuration info
        config["resizing"] = True
        config["img_channel"] = 1 #só usa um canal de cores pois está a trabalhar com imagens a preto e branco
        config["batch_size"] = 1  #processa umaimagem de cada vez
        config["num_multilabel"] = 5  # the number of label (channel last)

        # load image
        image = sitk.GetArrayFromImage(sitk.ReadImage(x_list)).astype('float32') #carrega a imagem de entrada em z, y, x
        if config["resizing"] == True:
            image = self.resize(image)
            img_shape = image.shape
        else:
            img_shape = image.shape

        # reshape image to 5D tensor 
        image = np.reshape(image, (config["batch_size"], img_shape[0], img_shape[1],img_shape[2], config["img_channel"]))  # batch, z ,y, x , ch


        n_image = (image-np.min(image))/(np.max(image)-np.min(image)) # normaliza valores de intensidade entre 0 e 1

        label = 0

        if y_list != 0: #modo de treino (=0 se inferencia)
            # load label        
            labels = sitk.GetArrayFromImage(sitk.ReadImage(y_list)).astype('float32') #carrega a magem, que tem todos os voxels categorizados

            if config["resizing"] == True:
                labels = self.resize(labels)
                lb_shape = labels.shape
            else:
                lb_shape = labels.shape

            onehot = to_categorical(labels) #converte os labels para one-hot encoding

            label = np.reshape(
                onehot, (config["batch_size"], lb_shape[0], lb_shape[1], lb_shape[2], config["num_multilabel"]))

        return n_image, label


#Força qualquer matriz 3D a caber numa caixa de tamanho fixo (por defeito 128x128x128)
    def resize(self, data, img_dep=128, img_cols=128, img_rows=128):
        resize_factor = (
            img_dep/data.shape[0], img_cols/data.shape[1], img_rows/data.shape[2])
        data = ndimage.zoom(data, resize_factor, order=0,
                            mode='constant', cval=0.0)
        return data

    # where_do_you_want_to_blur? ex) where = (1,1,1) -> blur(eyes, nose, ears)
    def Deidentification_image_nii(self, where, nfti_path, dest_path, verif_path, prefix, Model=model):
        '''
        where : list or tuple. Each position stands for eyes nose ears (eyes, nose, ears) 
                If the corresponding position is 1, de-identification process.
        model : Predictive model to be applied.
        '''
        config = dict()  # configuration info
        config["resizing"] = True
        config["input_shape"] = [128, 128, 128, 1]
        prefix += "_{}"

        try:
            # get affine and header of original image file.
            raw_img = nib.load(nfti_path)
            array_img = raw_img.get_fdata()  # image array
            original_shape = array_img.shape  # (x,y,z)
            thresh = threshold_triangle(array_img)

            # load prediction label
            image, label = self.load_batch(nfti_path)  # z, y, x
            results = model.model.predict(image)
            results = np.round(results)

            # preprocessing: Size recovery and transform onehot to labels number
            if config["resizing"] == True:
                results = self.onehot2label(results)
                # prediction results (batch size, dep, col ,row, ch) -> (dep, col ,row)
                results = np.reshape(results, config["input_shape"][0:3])
                results = self.resize(results,
                                      img_dep=original_shape[2],
                                      img_cols=original_shape[1],
                                      img_rows=original_shape[0])
                # except 0 label (blanck)
                results = to_categorical(results)

            else:
                results = results[0, ...]

            # search center by clustering
            boxes = self.bounding_box(results[..., 1:])

            #if len(boxes) != 6:
            #    raise Exception("Can not find all the parts of the face")

            # view label with .png
            if not os.path.isdir(verif_path):
                os.makedirs(verif_path)
            fileName = os.path.basename(dest_path)
            self.nifti_view_label(array_img, results, boxes, verif_path, fileName)

            # blur parts of face
            if where[1]:  # nose
                box = boxes[2]
                array_img = self.box_blur(array_img, box, 1, wth=1.33)

            # make outer contour for mini array.
            edge_img = self.outer_contour_3D(array_img, zoom=1)

            if where[0]:  # eyes

                box = boxes[0]  # eye
                array_img = self.surface_blur(
                    array_img, edge_img, box, wth=1.5, dep=3, option=1)

                box = boxes[1]  # eye
                array_img = self.surface_blur(
                    array_img, edge_img, box, wth=1.5, dep=3, option=1)

            if where[2]:  # ears
                '''
                In order not to see the outline of the ear due to external noise,
                fill the area of the ear with similar noise
                '''
                ear_results = results[...,3]
                border = self.box_blur(np.ones(array_img.shape), boxes[3], 1) #'box_blur' function is based on array_img.shape (nibabel liabrary)
                border = self.box_blur(border, boxes[4], 1)
                border = 1-border
                ear_results = border*ear_results.T
                
                noise = np.random.rand(*original_shape)*thresh*0.8 
                array_img[ear_results == 1] = noise[ear_results == 1]

            if where[3] : # mouth
            
                mouth_results = results[...,4] 
                border = self.box_blur(np.ones(array_img.shape),boxes[5],1) #'box_blur' function is based on array_img.shape (nibabel liabrary)
                border = 1-border
                if where[1] == False: # If you want to preserve the nose
                        border = self.box_blur(border,boxes[2],1, wth=1.5)
                        
                mouth_results = border*mouth_results.T
                
                threshold = np.max(ndimage.gaussian_filter(array_img[mouth_results==1],sigma=3))
                array_img[mouth_results==1] = threshold
                

            array_img = np.round(array_img)
            array_img = np.array(array_img, dtype='int32')

            nib.save(nib.Nifti1Image(array_img, raw_img.affine, raw_img.header),
                     os.path.join(os.path.dirname(dest_path), prefix.format(os.path.basename(nfti_path))))

            return {"success": True, "msg": ""}
        except Exception as ex:
            print(ex)
            return {"success": False, "msg": str(ex)}
