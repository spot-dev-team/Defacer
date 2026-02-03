import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure, binary_erosion, center_of_mass

class SpotSegmenter:
    def __init__(self, face_mask_path, internal_mask_path, output_dir):
        self.face_path = face_mask_path
        self.int_path = internal_mask_path
        self.output_dir = output_dir
        
        print(f"[Spot] A carregar dados...")
        self.face_img = nib.load(self.face_path)
        self.int_img = nib.load(self.int_path)
        
        self.face_data = self.face_img.get_fdata()
        self.int_data = self.int_img.get_fdata()
        self.affine = self.face_img.affine
        self.voxel_sizes = nib.affines.voxel_sizes(self.affine)
        
        self.masks = {
            'eyes': None,
            'ears': None,
            'face_main': None,
            'nose': None,
            'mouth': None 
        }
        
        # Variáveis de navegação espacial
        self.nose_tip_coords = None
        self.z_direction = -1 # -1 se Z diminui para baixo, 1 se aumenta
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process(self):
        print("[Spot] Iniciando Pipeline v4 (Boca Relativa)...")
        
        # 1. Olhos
        self._extract_eyes()
        
        # 2. Orelhas (Topologia)
        self._separate_ears_from_face()
        
        if self.masks['face_main'] is not None:
            # 3. Nariz (Túnel v3)
            self._find_nose_tip()
            self._extract_nose_tunnel()
            
            # 4. Boca (Relativa ao Nariz)
            self._extract_mouth_relative() # <--- NOVA LÓGICA
            
            # Limpeza Final: Garantir que não há sobreposições
            # A 'face_main' fica apenas com as bochechas e testa
            print(" -> A limpar sobreposições finais...")
            remove_mask = (self.masks['nose'] == 1) | (self.masks['mouth'] == 1)
            self.masks['face_main'] = np.where(
                remove_mask, 
                0, 
                self.masks['face_main']
            )
        
        self._save_outputs()

    def _extract_eyes(self):
        print(" -> A extrair Olhos...")
        self.masks['eyes'] = np.where(self.int_data == 259, 1, 0).astype(np.uint8)

    def _separate_ears_from_face(self):
        print(" -> A separar Orelhas...")
        binary_face = np.where(self.face_data > 0, 1, 0).astype(np.uint8)
        struct = generate_binary_structure(3, 2)
        labeled_array, num_features = label(binary_face, structure=struct)
        
        if num_features < 2:
            eroded = binary_erosion(binary_face, iterations=2)
            labeled_array, num_features = label(eroded, structure=struct)

        sizes = np.bincount(labeled_array.ravel())
        sizes[0] = 0
        sorted_indices = np.argsort(sizes)[::-1]
        
        main_face_idx = sorted_indices[0]
        ear_indices = [idx for idx in sorted_indices[1:] if sizes[idx] > 50]
        
        self.masks['face_main'] = np.where(labeled_array == main_face_idx, 1, 0).astype(np.uint8)
        self.masks['ears'] = np.isin(labeled_array, ear_indices).astype(np.uint8)

    def _find_nose_tip(self):
        face_voxels = np.argwhere(self.masks['face_main'] == 1)
        if len(face_voxels) == 0: return
        
        # Assumindo Y (1) como Anterior-Posterior
        axis_ap = 1 
        tip_idx = np.argmax(face_voxels[:, axis_ap])
        self.nose_tip_coords = face_voxels[tip_idx]
        print(f"    Ponta do Nariz: {self.nose_tip_coords}")

    def _extract_nose_tunnel(self):
        print(" -> A recortar Nariz (Túnel Híbrido: BackWall + ConvexHull)...")
        if self.nose_tip_coords is None or self.masks['eyes'] is None: return
        
        # Importação necessária
        from skimage.morphology import convex_hull_image

        # 1. Calcular Orientação e Limites (Igual)
        eyes_center = center_of_mass(self.masks['eyes'])
        z_eyes = eyes_center[2]
        z_nose = self.nose_tip_coords[2]
        
        if z_eyes > z_nose:
            self.z_direction = -1 
        else:
            self.z_direction = 1

        z_top_voxel = int(z_eyes)
        
        # Raio 35mm (Mantém-se largo para apanhar as laterais)
        radius_mm = 35.0 
        radius_voxels_z = int(radius_mm / self.voxel_sizes[2])
        z_bottom_voxel = int(z_nose + (radius_voxels_z * self.z_direction))
        self.nose_z_bottom_limit = z_bottom_voxel 

        # 2. Construção do Cilindro Geométrico
        radius_voxels_x = int(radius_mm / self.voxel_sizes[0])
        x_center = self.nose_tip_coords[0]
        
        z_min = min(z_top_voxel, z_bottom_voxel)
        z_max = max(z_top_voxel, z_bottom_voxel)
        z_mid = (z_min + z_max) / 2
        z_radius = (z_max - z_min) / 2
        
        shape = self.face_data.shape
        x_grid, y_grid, z_grid = np.ogrid[:shape[0], :shape[1], :shape[2]]
        
        tunnel_geo_mask = (
            ((x_grid - x_center)**2 / radius_voxels_x**2) + 
            ((z_grid - z_mid)**2 / z_radius**2) <= 1
        )
        
        # 3. LÓGICA HÍBRIDA (Fatia a Fatia)
        # Inicializar com as dimensões CORRETAS (X, Y, Z) para evitar erro de broadcasting
        final_nose_mask = np.zeros(shape, dtype=np.uint8)
        
        print("    A gerar triângulo sólido limitado pela face...")
        
        y_indices = np.arange(shape[1]) # Grelha Y para calculos rápidos

        for z in range(z_min, z_max):
            # A. Obter a pele dentro do cilindro nesta fatia
            skin_slice = self.masks['face_main'][:, :, z]
            geo_slice = tunnel_geo_mask[:, :, z]
            intersection = skin_slice & geo_slice
            
            if not np.any(intersection): continue
                
            # B. Encontrar a "Parede de Fundo" (Back Wall)
            coords = np.argwhere(intersection)
            if len(coords) == 0: continue

            # Encontrar os pontos mais recuados (menor Y) dos lados esquerdo e direito
            # para garantir que não cortamos a bochecha.
            x_values = coords[:, 0]
            y_values = coords[:, 1]
            
            # Ponto mais à esquerda e mais à direita
            idx_min_x = np.argmin(x_values)
            idx_max_x = np.argmax(x_values)
            
            # O limite é o ponto de inserção mais "fundo" (menor Y)
            y_back_limit = min(y_values[idx_min_x], y_values[idx_max_x])
            
            # C. RECORTAR PRIMEIRO (Clip)
            # Criamos uma máscara da pele que existe APENAS à frente da linha
            # Isto remove a parte de trás do cilindro que criava o retângulo
            forward_mask = y_indices[None, :] > y_back_limit # (1, Y)
            nose_tip_skin = intersection & forward_mask # Interseção da pele com a frente
            
            # D. ENCHER DEPOIS (Convex Hull)
            # Agora aplicamos o Convex Hull APENAS na ponta recortada.
            # O Convex Hull vai ligar a ponta do nariz à linha de corte (y_back_limit),
            # criando o triângulo perfeito e preenchendo o "vácuo".
            if np.any(nose_tip_skin):
                solid_triangle = convex_hull_image(nose_tip_skin)
                final_nose_mask[:, :, z] = solid_triangle.astype(np.uint8)

        self.masks['nose'] = final_nose_mask



    def _extract_mouth_relative(self):
        """
        Lógica:
        1. Pegar no limite inferior do nariz.
        2. Dar um gap (ex: 4mm).
        3. Tudo o que for face abaixo disso = Boca (inclui queixo).
        """
        print(" -> A recortar Boca (Relativa ao Nariz)...")
        if self.masks['nose'] is None: return

        # Configurações
        gap_mm = 4.0
        gap_voxels = int(gap_mm / self.voxel_sizes[2])
        
        # Ponto de corte inicial
        z_cut_start = self.nose_z_bottom_limit + (gap_voxels * self.z_direction)
        
        print(f"    Corte da boca inicia em Z={z_cut_start} (Gap de {gap_mm}mm)")
        
        shape = self.face_data.shape
        _, _, z_grid = np.ogrid[:shape[0], :shape[1], :shape[2]]
        
        # Criar máscara de corte planar
        if self.z_direction == -1:
            # Baixo é menor que Z
            planar_mask = z_grid < z_cut_start
        else:
            # Baixo é maior que Z
            planar_mask = z_grid > z_cut_start
            
        # Interseção: O que é 'Face Principal' E está 'Abaixo do Gap'
        self.masks['mouth'] = (self.masks['face_main'] * planar_mask).astype(np.uint8)

    def _create_combined_labelmap(self):
        print(" -> A criar LabelMap Final Combinado (0-5)...")
        # Criar matriz vazia com as mesmas dimensões
        combined_data = np.zeros(self.face_data.shape, dtype=np.uint8)
        
        # Ordem de pintura (do menos importante para o mais importante)
        # Assim garantimos que se houver sobreposição, o órgão mais importante ganha.
        
        # 1. Face Base (Valor 1)
        if self.masks['face_main'] is not None:
            combined_data[self.masks['face_main'] == 1] = 1
            
        # 2. Orelhas (Valor 2)
        if self.masks['ears'] is not None:
            combined_data[self.masks['ears'] == 1] = 2
            
        # 3. Boca (Valor 3)
        if self.masks['mouth'] is not None:
            combined_data[self.masks['mouth'] == 1] = 3
            
        # 4. Nariz (Valor 4)
        if self.masks['nose'] is not None:
            combined_data[self.masks['nose'] == 1] = 4
            
        # 5. Olhos (Valor 5) - Prioridade Máxima
        if self.masks['eyes'] is not None:
            combined_data[self.masks['eyes'] == 1] = 5

        # Guardar
        img = nib.Nifti1Image(combined_data, self.affine, self.face_img.header)
        out_path = os.path.join(self.output_dir, "spot_final_combined.nii.gz")
        nib.save(img, out_path)
        print(f"    [SUPER FICHEIRO] Guardado em: {out_path}")
    


    def _save_outputs(self):
        print(" -> A gravar ficheiros...")
        for name, data in self.masks.items():
            if data is not None:
                img = nib.Nifti1Image(data, self.affine, self.face_img.header)
                out_path = os.path.join(self.output_dir, f"spot_{name}.nii.gz")
                nib.save(img, out_path)
                print(f"    Guardado: {out_path}")
        self._create_combined_labelmap()

# --- EXECUÇÃO ---
if __name__ == "__main__":
    # Ajusta estes caminhos para o teu teste
    # No Windows usa r"C:\..." ou barras duplas \\
    
    BASE_DIR = r"C:\Tese\Datasets\Defacer\ADNI_Nifti_Single_Folder\Outputs_Spot"

    # Inputs
    FACE_FILE = os.path.join(BASE_DIR, "002_S_0413_MPRAGE_SENSE_20061115141346_501_mask.nii.gz") # <--- CONFIRMA O NOME
    INT_FILE = os.path.join(BASE_DIR, "002_S_0413_MPRAGE_SENSE_20061115141346_501_olhos_raw.nii.gz") # <--- CONFIRMA O NOME
    
    # Output
    OUT_DIR = os.path.join(BASE_DIR, "Spot_Final_Masks\\teste12")
    
    # Run
    segmenter = SpotSegmenter(FACE_FILE, INT_FILE, OUT_DIR)
    segmenter.process() 