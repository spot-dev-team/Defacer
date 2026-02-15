# vai dividir a segmntação original do mideface sem transformar nada das mascaras em si, sem as expandir, nada

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure, center_of_mass
import matplotlib.pyplot as plt
from matplotlib import colors

# --- CONFIGURAÇÃO ---
BASE_DIR = r"D:\Tese_BDs\Defacer\ADNI_Structured"

# =========================================================================
# CLASSE: CLEAN SEGMENTER (Apenas Cortes, Sem Transformações)
# =========================================================================
# =========================================================================
# CLASSE: CLEAN SEGMENTER (Versão Híbrida: Gap vs No-Gap)
# =========================================================================
# =========================================================================
# CLASSE: CLEAN SEGMENTER (Algoritmo Inteligente: Subnasale Detection)
# =========================================================================
class CleanSegmenter:
    def __init__(self, face_path, int_path):
        self.face_img = nib.load(face_path)
        self.face_img = nib.as_closest_canonical(self.face_img)
        self.face_data = self.face_img.get_fdata() > 0 # Binário
        
        self.int_img = nib.load(int_path)
        self.int_img = nib.as_closest_canonical(self.int_img)
        self.int_data = self.int_img.get_fdata()
        
        self.affine = self.face_img.affine
        self.voxel_sizes = nib.affines.voxel_sizes(self.affine)
        
        self.masks = {'eyes': None, 'ears': None, 'nose': None, 'mouth': None, 'face_rest': None}

    def process(self):
        # 1. OLHOS
        self.masks['eyes'] = (self.int_data == 259).astype(np.uint8)
        
        # 2. ORELHAS (Topologia)
        struct = generate_binary_structure(3, 2)
        lbl_array, n_feats = label(self.face_data, structure=struct)
        sizes = np.bincount(lbl_array.ravel())
        sizes[0] = 0
        sorted_idx = np.argsort(sizes)[::-1]
        
        face_main_island = (lbl_array == sorted_idx[0])
        ear_indices = [i for i in sorted_idx[1:] if sizes[i] > 50]
        self.masks['ears'] = np.isin(lbl_array, ear_indices).astype(np.uint8)
        
        # 3. LOGICA DO NARIZ INTELIGENTE
        face_voxels = np.argwhere(face_main_island)
        if len(face_voxels) == 0: return 
        
        # Encontrar a Ponta
        tip_idx = np.argmax(face_voxels[:, 1])
        nose_tip = face_voxels[tip_idx] # [x, y, z]
        
        # Direção Z (Olhos -> Nariz)
        if np.any(self.masks['eyes']):
            eyes_center = center_of_mass(self.masks['eyes'])
            z_eyes = eyes_center[2]
        else:
            z_eyes = nose_tip[2] + 20 
        
        z_nose = nose_tip[2]
        z_dir = -1 if z_eyes > z_nose else 1 # -1 se descermos no array para ir para o queixo
        
        # --- A TUA IDEIA: DETETAR A "QUEDA" (Subnasale) ---
        z_bottom_smart = self._find_smart_nose_bottom(face_main_island, nose_tip, z_dir)
        
        # Criar a Máscara do Cilindro (Largura 35mm, Altura Dinâmica)
        rad_mm = 35.0
        rad_x = int(rad_mm / self.voxel_sizes[0])
        
        # O cilindro agora vai da linha dos olhos até ao ponto inteligente que achámos
        z_top = int(z_eyes)
        z_min, z_max = min(z_top, z_bottom_smart), max(z_top, z_bottom_smart)
        z_mid = (z_min + z_max) / 2
        z_rad_val = (z_max - z_min) / 2
        
        shape = self.face_data.shape
        x_g, _, z_g = np.ogrid[:shape[0], :shape[1], :shape[2]]
        
        cylinder_mask = (
            ((x_g - nose_tip[0])**2 / rad_x**2) + 
            ((z_g - z_mid)**2 / z_rad_val**2) <= 1
        )
        
        # O PLANO DE CORTE agora é o nosso ponto inteligente
        z_split = z_bottom_smart
        
        if z_dir == -1: 
            # Z diminui para baixo. Nariz é MAIOR que z_split. Boca é MENOR.
            mask_below = z_g < z_split
        else: 
            # Z aumenta para baixo. Nariz é MENOR que z_split. Boca é MAIOR.
            mask_below = z_g > z_split
            
        # NARIZ = Dentro do Cilindro E Acima do Corte Inteligente
        self.masks['nose'] = (face_main_island & cylinder_mask & ~mask_below).astype(np.uint8)
        
        # BOCA = Tudo o que sobra abaixo do corte inteligente
        self.masks['mouth'] = (face_main_island & mask_below).astype(np.uint8)
        
        # FACE REST
        used = (self.masks['nose'] | self.masks['mouth'] | self.masks['ears'])
        self.masks['face_rest'] = (face_main_island & ~used).astype(np.uint8)

    def _find_smart_nose_bottom(self, face_mask, nose_tip, z_dir):
        """
        Implementa a tua ideia: Percorre o perfil do nariz para baixo até encontrar
        o ponto onde ele recua ao máximo (Subnasale) antes do lábio.
        """
        max_search_mm = 20.0 # Procura no máximo 2cm abaixo da ponta
        max_search_vox = int(max_search_mm / self.voxel_sizes[2])
        
        start_z = nose_tip[2]
        current_min_y = nose_tip[1] # Começamos na ponta (Y máximo)
        best_z = start_z
        
        # Vamos iterar Z "para baixo" (direção da boca)
        for i in range(1, max_search_vox):
            z_curr = start_z + (i * z_dir)
            
            # Verificar limites da imagem
            if z_curr < 0 or z_curr >= face_mask.shape[2]: break
            
            # Obter a fatia da cara neste Z
            slice_mask = face_mask[:, :, z_curr]
            if not np.any(slice_mask): continue # Acabou a cara?
            
            # Encontrar o ponto mais à frente (Max Y) nesta fatia
            # Limitamos a busca em X para não apanhar bochechas laterais, só o centro
            center_x = nose_tip[0]
            margin_x = 10 # +/- 10 voxels à volta do centro
            x_min_s = max(0, center_x - margin_x)
            x_max_s = min(face_mask.shape[0], center_x + margin_x)
            
            # Recorte central da fatia
            center_strip = slice_mask[x_min_s:x_max_s, :]
            if not np.any(center_strip): continue
            
            coords = np.argwhere(center_strip)
            if len(coords) == 0: continue
            
            # O Y mais anterior nesta linha (lembra-te que coords[:,1] é o Y original)
            y_curr = np.max(coords[:, 1])
            
            # LÓGICA DO "MERGULHO":
            # O nariz vai "encolhendo" (Y diminui) à medida que descemos.
            # O Subnasale é o ponto MÍNIMO local de Y.
            # Se o Y começar a aumentar de novo, encontrámos o lábio superior!
            
            # Mas cuidado com ruído. Vamos simplesmente procurar o ponto mais "fundo" (menor Y)
            # dentro desta janela de 20mm.
            
            if y_curr < current_min_y:
                current_min_y = y_curr
                best_z = z_curr
            else:
                # Se o Y aumentou (lábio a sair), e já descemos pelo menos uns 5mm...
                # Podemos parar, achámos a inflexão!
                dist_mm = i * self.voxel_sizes[2]
                if dist_mm > 5.0 and y_curr > current_min_y + 1: # +1 voxel de tolerância
                    return best_z
                    
        return best_z

    def save_masks(self, out_dir):
        # 1. 4 CLASSES
        m4 = np.zeros(self.face_data.shape, dtype=np.uint8)
        if self.masks['face_rest'] is not None: m4[self.masks['face_rest']==1] = 1
        if self.masks['ears'] is not None: m4[self.masks['ears']==1] = 2
        if self.masks['mouth'] is not None: m4[self.masks['mouth']==1] = 3
        if self.masks['nose'] is not None: m4[self.masks['nose']==1] = 4
        if self.masks['eyes'] is not None: m4[self.masks['eyes']==1] = 5
        nib.save(nib.Nifti1Image(m4, self.affine, self.face_img.header), os.path.join(out_dir, "mask_4_classes.nii.gz"))
        
        # 2. 3 CLASSES
        m3 = np.zeros(self.face_data.shape, dtype=np.uint8)
        if self.masks['face_rest'] is not None: m3[self.masks['face_rest']==1] = 1
        if self.masks['ears'] is not None: m3[self.masks['ears']==1] = 2
        lower_face = (self.masks['mouth']==1) | (self.masks['nose']==1)
        m3[lower_face] = 3
        if self.masks['eyes'] is not None: m3[self.masks['eyes']==1] = 5
        nib.save(nib.Nifti1Image(m3, self.affine, self.face_img.header), os.path.join(out_dir, "mask_3_classes.nii.gz"))
        
        return m4 # Retorna a de 4 classes para o QA

# =========================================================================
# FUNÇÃO: GERAR QA
# =========================================================================
def create_qa(image_path, labels, save_path, fname):
    img = nib.load(image_path).get_fdata()
    
    cmap = colors.ListedColormap(['none', 'gray', 'purple', 'blue', 'yellow', 'red'])
    # 0=bg, 1=face, 2=ears, 3=mouth, 4=nose, 5=eyes
    norm = colors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    plt.suptitle(f"QA RAW: {fname}", fontsize=14)
    
    # Organs to show (Label, Name)
    organs = [(5, "Eyes"), (4, "Nose"), (2, "Ears"), (3, "Mouth")]
    
    for c, (lid, name) in enumerate(organs):
        # Encontrar melhor slice
        sums = np.sum(labels==lid, axis=tuple(i for i in range(3))) # Sum total
        if sums == 0: 
            z, x, y = [int(s/2) for s in img.shape]
        else:
            # Simple centroid approximation
            coords = np.argwhere(labels==lid)
            x, y, z = np.mean(coords, axis=0).astype(int)

        # Axial (Z)
        axes[0, c].imshow(np.rot90(img[:, :, z]), cmap='gray')
        axes[0, c].imshow(np.rot90(labels[:, :, z]), cmap=cmap, norm=norm, alpha=0.6)
        axes[0, c].set_title(f"{name} - Ax")
        axes[0, c].axis('off')
        
        # Sagital (X)
        axes[1, c].imshow(np.rot90(img[x, :, :]), cmap='gray')
        axes[1, c].imshow(np.rot90(labels[x, :, :]), cmap=cmap, norm=norm, alpha=0.6)
        axes[1, c].set_title(f"Sag")
        axes[1, c].axis('off')
        
        # Coronal (Y)
        axes[2, c].imshow(np.rot90(img[:, y, :]), cmap='gray')
        axes[2, c].imshow(np.rot90(labels[:, y, :]), cmap=cmap, norm=norm, alpha=0.6)
        axes[2, c].set_title(f"Cor")
        axes[2, c].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

# =========================================================================
# PIPELINE PRINCIPAL
# =========================================================================
def run_mask_division():
    subjects = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    print(f"A processar {len(subjects)} exames...")
    
    for i, subj in enumerate(subjects):
        subj_path = os.path.join(BASE_DIR, subj)
        
        # Inputs
        mid_dir = os.path.join(subj_path, "mid_output")
        face_f = os.path.join(mid_dir, "face_mask.nii.gz")
        eyes_f = os.path.join(mid_dir, "eyes_raw.nii.gz")
        
        # Raw Image (para o QA)
        raw_f = os.path.join(subj_path, "raw.nii.gz")
        if not os.path.exists(raw_f): # Fallback
             try: raw_f = next(os.path.join(subj_path, f) for f in os.listdir(subj_path) if f.endswith(".nii.gz") and "mask" not in f)
             except: continue

        # Output folder
        out_dir = os.path.join(subj_path, "training_masks")
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        
        if os.path.exists(face_f) and os.path.exists(eyes_f):
            print(f"[{i+1}] A gerar máscaras limpas: {subj}")
            try:
                # 1. Segmentar
                seg = CleanSegmenter(face_f, eyes_f)
                seg.process()
                
                # 2. Guardar NIfTI
                mask_data = seg.save_masks(out_dir)
                
                # 3. Gerar QA
                qa_path = os.path.join(out_dir, "QA_Validation.png")
                create_qa(raw_f, mask_data, qa_path, subj)
                
            except Exception as e:
                print(f"ERROR: {e}")
        else:
            print(f"[{i+1}] SKIP (Faltam inputs): {subj}")

if __name__ == "__main__":
    run_mask_division()