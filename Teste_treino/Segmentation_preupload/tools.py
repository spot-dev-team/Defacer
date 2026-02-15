import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure, binary_erosion, center_of_mass, binary_dilation, gaussian_filter, binary_opening, binary_closing, binary_fill_holes, shift
from skimage.morphology import convex_hull_image
from skimage.filters import threshold_triangle
import matplotlib.pyplot as plt
from matplotlib import colors

# =========================================================================
# CLASSE 1: SEGMENTADOR (Gera as máscaras geométricas precisas)
# =========================================================================
class SpotSegmenter:
    def __init__(self, face_mask_path, internal_mask_path, output_dir):
        self.face_img = nib.load(face_mask_path)
        self.face_img = nib.as_closest_canonical(self.face_img) # Normalização vital
        
        self.int_img = nib.load(internal_mask_path)
        self.int_img = nib.as_closest_canonical(self.int_img)

        self.face_data = self.face_img.get_fdata()
        self.int_data = self.int_img.get_fdata()
        self.affine = self.face_img.affine
        self.voxel_sizes = nib.affines.voxel_sizes(self.affine)
        self.output_dir = output_dir
        
        self.masks = {'eyes': None, 'ears': None, 'face_main': None, 'nose': None, 'mouth': None}
        self.nose_tip_coords = None
        self.nose_z_bottom_limit = 0
        self.z_direction = -1

    def process(self):
        # 1. Olhos (Label 259 do Samseg)
        self.masks['eyes'] = np.where(self.int_data == 259, 1, 0).astype(np.uint8)
        
        # 2. Orelhas & Face Main
        binary_face = np.where(self.face_data > 0, 1, 0).astype(np.uint8)
        struct = generate_binary_structure(3, 2)
        labeled_array, num_features = label(binary_face, structure=struct)
        if num_features < 2: # Se estiver tudo ligado, tenta erodir para separar
            eroded = binary_erosion(binary_face, iterations=2)
            labeled_array, num_features = label(eroded, structure=struct)
        
        sizes = np.bincount(labeled_array.ravel())
        sizes[0] = 0
        sorted_indices = np.argsort(sizes)[::-1]
        
        self.masks['face_main'] = np.where(labeled_array == sorted_indices[0], 1, 0).astype(np.uint8)
        # Orelhas são ilhas pequenas (mas não minúsculas)
        ear_indices = [idx for idx in sorted_indices[1:] if sizes[idx] > 50]
        self.masks['ears'] = np.isin(labeled_array, ear_indices).astype(np.uint8)

        # 3. Nariz & Boca
        if np.any(self.masks['face_main']):
            self._find_nose_tip()
            self._extract_nose_tunnel() # Versão Híbrida BackWall
            self._extract_mouth_relative()
            
            # Limpeza final de sobreposições
            remove_mask = (self.masks['nose'] == 1) | (self.masks['mouth'] == 1)
            self.masks['face_main'] = np.where(remove_mask, 0, self.masks['face_main'])

    def _find_nose_tip(self):
        face_voxels = np.argwhere(self.masks['face_main'] == 1)
        if len(face_voxels) == 0: return
        # Assumindo Y (1) como Anterior-Posterior
        tip_idx = np.argmax(face_voxels[:, 1])
        self.nose_tip_coords = face_voxels[tip_idx]

    def _extract_nose_tunnel(self):
        # LÓGICA HÍBRIDA: BackWall + Convex Hull
        if self.nose_tip_coords is None or not np.any(self.masks['eyes']): return

        eyes_center = center_of_mass(self.masks['eyes'])
        z_eyes, z_nose = eyes_center[2], self.nose_tip_coords[2]
        self.z_direction = -1 if z_eyes > z_nose else 1

        radius_mm = 35.0 # Largo para apanhar bochechas
        rad_z = int(radius_mm / self.voxel_sizes[2])
        rad_x = int(radius_mm / self.voxel_sizes[0])
        
        z_top = int(z_eyes)
        z_bottom = int(z_nose + (rad_z * self.z_direction))
        self.nose_z_bottom_limit = z_bottom
        
        z_min, z_max = min(z_top, z_bottom), max(z_top, z_bottom)
        z_mid = (z_min + z_max) / 2
        z_radius = (z_max - z_min) / 2
        
        shape = self.face_data.shape
        x_grid, y_grid, z_grid = np.ogrid[:shape[0], :shape[1], :shape[2]]
        
        # Cilindro Geométrico Base
        tunnel_geo = (((x_grid - self.nose_tip_coords[0])**2 / rad_x**2) + 
                      ((z_grid - z_mid)**2 / z_radius**2) <= 1)
        
        final_nose = np.zeros(shape, dtype=np.uint8)
        y_indices = np.arange(shape[1])

        for z in range(z_min, z_max):
            skin_slice = self.masks['face_main'][:, :, z]
            geo_slice = tunnel_geo[:, :, z]
            intersection = skin_slice & geo_slice
            
            if not np.any(intersection): continue
            coords = np.argwhere(intersection)
            if len(coords) == 0: continue

            x_vals, y_vals = coords[:, 0], coords[:, 1]
            idx_min_x, idx_max_x = np.argmin(x_vals), np.argmax(x_vals)
            
            # Back Wall: Ponto mais recuado
            y_back_limit = min(y_vals[idx_min_x], y_vals[idx_max_x])
            
            # Recortar frente
            forward_mask = y_indices[None, :] > y_back_limit
            nose_tip_skin = intersection & forward_mask
            
            # Encher (Convex Hull)
            if np.any(nose_tip_skin):
                final_nose[:, :, z] = convex_hull_image(nose_tip_skin).astype(np.uint8)
        
        self.masks['nose'] = final_nose

    def _extract_mouth_relative(self):
        if self.masks['nose'] is None: return
        gap_voxels = int(4.0 / self.voxel_sizes[2]) # 4mm gap
        z_cut = self.nose_z_bottom_limit + (gap_voxels * self.z_direction)
        
        _, _, z_grid = np.ogrid[:self.face_data.shape[0], :self.face_data.shape[1], :self.face_data.shape[2]]
        planar_mask = z_grid < z_cut if self.z_direction == -1 else z_grid > z_cut
        self.masks['mouth'] = (self.masks['face_main'] * planar_mask).astype(np.uint8)

    def save_final_mask(self):
        # Gera o ficheiro combinado (1-5)
        combined = np.zeros(self.face_data.shape, dtype=np.uint8)
        # Prioridades
        if self.masks['face_main'] is not None: combined[self.masks['face_main']==1] = 1
        if self.masks['ears'] is not None: combined[self.masks['ears']==1] = 2 # Ears
        if self.masks['mouth'] is not None: combined[self.masks['mouth']==1] = 4 # Boca (Spot v4 code uses 4 for mouth sometimes, adapting to your prev logic: 3=Ears/Mouth?) 
        # Vamos padronizar: 1=Face, 2=Orelhas, 3=Boca, 4=Nariz, 5=Olhos
        if self.masks['mouth'] is not None: combined[self.masks['mouth']==1] = 3
        if self.masks['nose'] is not None: combined[self.masks['nose']==1] = 4
        if self.masks['eyes'] is not None: combined[self.masks['eyes']==1] = 5
        
        out_path = os.path.join(self.output_dir, "spot_final_combined.nii.gz")
        nib.save(nib.Nifti1Image(combined, self.affine, self.face_img.header), out_path)
        return out_path

# =========================================================================
# CLASSE 2: DEFACER (Aplica o blur/flattening)
# =========================================================================
class SpotDefacer:
    def Deidentification_image_nii_SPOT(self, where, nfti_path, spot_mask_path, dest_path, verif_path, prefix):
        # where = [Eyes, Nose, Ears, Mouth]
        try:
            raw_img = nib.load(nfti_path)
            raw_img = nib.as_closest_canonical(raw_img)
            array_img = raw_img.get_fdata()
            
            spot_data = nib.load(spot_mask_path).get_fdata() # Já canonizado pelo Segmenter
            
            # Remap labels to internal logic: 1=Eyes, 2=Nose, 3=Ears, 4=Mouth
            remapped = np.zeros_like(spot_data)
            remapped[spot_data == 5] = 1 # Olhos
            remapped[spot_data == 4] = 2 # Nariz
            remapped[spot_data == 2] = 3 # Orelhas
            remapped[spot_data == 3] = 4 # Boca

            struct = generate_binary_structure(3, 1)
            
            # --- PREPARAÇÃO MÁSCARAS ---
            # Nariz
            if where[1] and np.any(remapped==2): 
                dilated = binary_dilation(remapped==2, structure=struct, iterations=1)
                remapped[dilated] = 2
            # Orelhas
            if where[2] and np.any(remapped==3):
                dilated = binary_dilation(remapped==3, structure=struct, iterations=2)
                remapped[dilated] = 3
            # Boca
            if where[3] and np.any(remapped==4):
                dilated = binary_dilation(remapped==4, structure=struct, iterations=2)
                remapped[dilated] = 4
            # Olhos (Expansão Direcional)
            if where[0] and np.any(remapped==1):
                mask_eyes = remapped == 1
                mask_eyes = binary_closing(binary_opening(mask_eyes, iterations=1), iterations=3)
                mask_raw = binary_fill_holes(mask_eyes)
                
                # Vetor e Shift
                img_center = np.array(array_img.shape) / 2.0
                eye_center = center_of_mass(mask_raw)
                d_vec = (eye_center - img_center) / np.linalg.norm(eye_center - img_center)
                
                dom_axis = np.argmax(np.abs(d_vec))
                is_pos = d_vec[dom_axis] > 0
                eye_idx = np.where(mask_raw)
                post_limit = np.min(eye_idx[dom_axis]) if is_pos else np.max(eye_idx[dom_axis])
                
                mask_base = binary_dilation(mask_raw, structure=struct, iterations=2)
                cum_mask = mask_base.copy()
                
                for i in range(1, 11):
                    shifted = shift(mask_base.astype(float), d_vec * i * 1.5, order=0)
                    cum_mask = cum_mask | (shifted > 0.5)
                
                # Trava Segurança
                slicer = [slice(None)]*3
                if is_pos: slicer[dom_axis] = slice(0, post_limit)
                else: slicer[dom_axis] = slice(post_limit+1, None)
                safety = np.ones_like(cum_mask, dtype=bool)
                safety[tuple(slicer)] = False
                
                mask_eyes_final = cum_mask & safety
                remapped[mask_eyes_final] = 1

            # --- QA ---
            if not os.path.isdir(verif_path): os.makedirs(verif_path)
            self._generate_qa(array_img, remapped, verif_path, os.path.basename(dest_path))

            # --- APLICAÇÃO ---
            # Nariz (Wipe)
            if where[1]: array_img[remapped==2] = 0
            # Orelhas (Noise)
            if where[2]: 
                try: thresh = threshold_triangle(array_img)
                except: thresh = np.mean(array_img)*0.1
                noise = np.random.rand(*array_img.shape) * thresh * 0.8
                array_img[remapped==3] = noise[remapped==3]
            # Olhos (Flatten)
            if where[0]:
                mask = (remapped==1)
                if np.any(mask):
                    blurred = gaussian_filter(array_img, sigma=3)
                    # Amostrar frente
                    front_s = [slice(None)]*3
                    # Lógica simplificada de frente para sampling
                    if is_pos: front_s[dom_axis] = slice(post_limit+5, None)
                    else: front_s[dom_axis] = slice(0, post_limit-5)
                    mask_f = mask.copy()
                    mask_f[tuple(front_s)] = False
                    mask_f = mask & ~mask_f
                    
                    val = np.max(blurred[mask_f]) if np.any(mask_f) else np.max(blurred[mask])
                    array_img[mask] = val
            # Boca (Flatten)
            if where[3] and np.any(remapped==4):
                val = np.max(gaussian_filter(array_img, sigma=3)[remapped==4])
                array_img[remapped==4] = val

            # Guardar
            final_img = nib.Nifti1Image(array_img, raw_img.affine, raw_img.header)
            nib.save(final_img, dest_path)
            return True

        except Exception as e:
            print(f"Erro Defacer: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_qa(self, image, labels, save_path, fname):
        # Versão simplificada do MultiView para caber aqui
        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        cmap = colors.ListedColormap(['red', 'purple', 'blue', 'yellow'])
        bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        organs = [(1,"Eyes"), (2,"Nose"), (3,"Ears"), (4,"Mouth")]
        for c, (lid, name) in enumerate(organs):
            for r, axis in enumerate([2, 0, 1]): # Ax, Sag, Cor
                sums = np.sum(labels==lid, axis=tuple(i for i in range(3) if i!=axis))
                sl = np.argmax(sums) if np.sum(sums)>0 else int(image.shape[axis]/2)
                
                if axis==0: sl_img, sl_msk = image[sl,:,:], labels[sl,:,:]
                elif axis==1: sl_img, sl_msk = image[:,sl,:], labels[:,sl,:]
                else: sl_img, sl_msk = image[:,:,sl], labels[:,:,sl]
                
                axes[r,c].imshow(np.rot90(sl_img), cmap='gray')
                axes[r,c].imshow(np.ma.masked_where(np.rot90(sl_msk)!=lid, np.rot90(sl_msk)), cmap=cmap, norm=norm, alpha=0.7)
                axes[r,c].axis('off')
        
        plt.savefig(os.path.join(save_path, f"QA_{fname}.png"))
        plt.close()