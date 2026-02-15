import os
import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import colors
from scipy import ndimage

# --- IMPORTS ESPECÍFICOS ---
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from tensorflow.keras import backend as K

# ==============================================================================
# CLASSE InstanceNormalization (ORIGINAL)
# ==============================================================================
class InstanceNormalization(Layer):
    def __init__(self, axis=None, epsilon=1e-3, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones', **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)

    def build(self, input_shape):
        if self.axis is None: shape = (1,)
        else: shape = (input_shape[self.axis],)
        if self.scale: self.gamma = self.add_weight(shape=shape, name='gamma', initializer=self.gamma_initializer)
        if self.center: self.beta = self.add_weight(shape=shape, name='beta', initializer=self.beta_initializer)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))
        if self.axis is not None: del reduction_axes[self.axis]
        del reduction_axes[0]
        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev
        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None: broadcast_shape[self.axis] = input_shape[self.axis]
        if self.scale: normed = normed * K.reshape(self.gamma, broadcast_shape)
        if self.center: normed = normed + K.reshape(self.beta, broadcast_shape)
        return normed
    
    def get_config(self):
        config = {'axis': self.axis, 'epsilon': self.epsilon}
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
MODEL_PATH = "modelo_final_augmented.h5" 
OUTPUT_DIR = "inference_results"

TEST_CASES = [
    {
        "id": "ADNI_Treino",
        "path": r"C:\Tese\ADNI_Lite_Cluster\002_S_0413__2006-11-15_14_23_260__I30119",
        "has_mask": True
    },
    {
        "id": "IXI_Teste",
        "path": r"C:\Tese\IXI_Structured\IXI131-HH-1527-T1",
        "has_mask": False 
    }
]

# ==============================================================================
# FUNÇÕES DE PROCESSAMENTO
# ==============================================================================
def robust_resize(volume, dim=(128,128,128), order=1):
    """Reduz para 128x128x128 para entrar na IA"""
    if len(volume.shape) == 4: volume = volume[:, :, :, 0]
    min_val, max_val = np.min(volume), np.max(volume)
    volume = (volume - min_val) / (max_val - min_val + 1e-8)
    factors = (dim[0]/volume.shape[0], dim[1]/volume.shape[1], dim[2]/volume.shape[2])
    vol_res = ndimage.zoom(volume, factors, order=order)
    final_vol = np.zeros(dim)
    d0, d1, d2 = min(dim[0], vol_res.shape[0]), min(dim[1], vol_res.shape[1]), min(dim[2], vol_res.shape[2])
    start0, start1, start2 = (dim[0]-d0)//2, (dim[1]-d1)//2, (dim[2]-d2)//2
    final_vol[start0:start0+d0, start1:start1+d1, start2:start2+d2] = vol_res[:d0, :d1, :d2]
    return final_vol

def resize_back_to_original(prediction_128, original_shape):
    """
    Estica a máscara 128x128x128 de volta para o tamanho original (ex: 256x256x160).
    Usa interpolação 'nearest' (order=0) para manter os labels inteiros (1,2,3...).
    """
    # Remover padding (inverso do robust_resize)
    # Assumimos que o robust_resize centrou a imagem.
    
    # Fatores de escala aproximados
    factors = (original_shape[0]/128, original_shape[1]/128, original_shape[2]/128)
    
    # Zoom direto (Método Simplificado mas eficaz para overlays rápidos)
    # Nota: Para precisão milimétrica, teriamos de guardar o padding exato usado na ida.
    # Como o robust_resize centra, o zoom direto costuma funcionar bem se a proporção for mantida.
    
    resized = ndimage.zoom(prediction_128, factors, order=0) # order=0 para labels!
    
    # Ajuste fino de dimensões (pode falhar por 1 ou 2 pixeis devido a arredondamento)
    final = np.zeros(original_shape, dtype=np.uint8)
    
    # Crop ou Pad para caber exatamente
    d0 = min(resized.shape[0], original_shape[0])
    d1 = min(resized.shape[1], original_shape[1])
    d2 = min(resized.shape[2], original_shape[2])
    
    # Centrar
    s0 = (original_shape[0] - d0) // 2
    s1 = (original_shape[1] - d1) // 2
    s2 = (original_shape[2] - d2) // 2
    
    # Centrar origem
    r0 = (resized.shape[0] - d0) // 2
    r1 = (resized.shape[1] - d1) // 2
    r2 = (resized.shape[2] - d2) // 2
    
    final[s0:s0+d0, s1:s1+d1, s2:s2+d2] = resized[r0:r0+d0, r1:r1+d1, r2:r2+d2]
    
    return final

def dice_score(y_true, y_pred): return 0
def dice_loss(y_true, y_pred): return 0

# ==============================================================================
# VISUALIZADOR
# ==============================================================================
def visualize_interactive(img_vol, pred_vol, gt_vol, title):
    # Cores: 0=Fundo, 1=Pele, 2=Orelhas, 3=Boca, 4=Nariz, 5=Olhos
    # 'gray' para a pele para não distrair muito
    cmap = colors.ListedColormap(['black', 'gray', 'purple', 'blue', 'orange', 'red'])
    
    # Fronteiras para 6 classes (0 a 5)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    initial_slice = 64
    has_gt = gt_vol is not None
    cols = 3 if has_gt else 2
    
    fig, axes = plt.subplots(1, cols, figsize=(15, 6))
    plt.subplots_adjust(bottom=0.20)
    fig.suptitle(f"{title} (Use Slider)", fontsize=16)

    def update_plot(val):
        idx = int(val) # Cast para int
        for ax in axes: ax.clear()
        
        # MRI
        slice_img = np.rot90(img_vol[:, idx, :])
        axes[0].imshow(slice_img, cmap='gray')
        axes[0].set_title(f"MRI Slice {idx}")
        axes[0].axis('off')
        
        # Pred
        slice_pred = np.rot90(pred_vol[:, idx, :])
        axes[1].imshow(slice_img, cmap='gray', alpha=0.6)
        axes[1].imshow(slice_pred, cmap=cmap, norm=norm, alpha=0.5)
        axes[1].set_title("Previsão Spot")
        axes[1].axis('off')
        
        # GT
        if has_gt:
            slice_gt = np.rot90(gt_vol[:, idx, :])
            axes[2].imshow(slice_img, cmap='gray', alpha=0.6)
            axes[2].imshow(slice_gt, cmap=cmap, norm=norm, alpha=0.5)
            axes[2].set_title("Ground Truth")
            axes[2].axis('off')
            
        fig.canvas.draw_idle()

    update_plot(initial_slice)
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, 127, valinit=initial_slice, valstep=1)
    slider.on_changed(update_plot)
    plt.show()

# ==============================================================================
# MAIN
# ==============================================================================
def run_pipeline():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRO: Modelo não encontrado em {MODEL_PATH}")
        return

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print("1. A carregar o modelo...")
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects={'dice_loss': dice_loss, 'dice_score': dice_score, 'InstanceNormalization': InstanceNormalization}
    )

    for case in TEST_CASES:
        print(f"\n--- A processar: {case['id']} ---")
        folder = case['path']
        
        raw_path = os.path.join(folder, "raw.nii.gz")
        if not os.path.exists(raw_path):
            print(f"⚠️ Raw file não encontrado em {folder}")
            continue

        # 1. Carregar Original
        img_obj = nib.load(raw_path)
        img_data = img_obj.get_fdata()
        original_affine = img_obj.affine # Guardamos a posição original no mundo
        original_shape = img_data.shape
        print(f" -> Dimensões Originais: {original_shape}")
        
        # 2. Carregar GT (se houver)
        mask_data = None
        mask_path = os.path.join(folder, "training_masks", "mask_4_classes.nii.gz")
        if os.path.exists(mask_path):
            mask_data = nib.load(mask_path).get_fdata()

        # 3. Resize & Predict (128)
        print(" -> A redimensionar para 128...")
        img_resized = robust_resize(img_data, order=1)
        
        img_input = np.expand_dims(img_resized, axis=[0, -1])
        pred_prob = model.predict(img_input, verbose=0)
        pred_mask_128 = np.argmax(pred_prob[0], axis=-1)

        # ------------------------------------------------------------------
        # GUARDAR COM GEOMETRIA CORRETA (PARA O SLICER)
        # ------------------------------------------------------------------
        print(f" -> A redimensionar de volta para {original_shape}...")
        
        # Estica a previsão 128 de volta para o tamanho do raw
        pred_full_size = resize_back_to_original(pred_mask_128, original_shape)
        
        # Guarda usando o Header e Affine ORIGINAIS
        final_nifti = nib.Nifti1Image(pred_full_size.astype(np.uint8), original_affine, img_obj.header)
        
        save_path = os.path.join(OUTPUT_DIR, f"{case['id']}_prediction_FULL.nii.gz")
        nib.save(final_nifti, save_path)
        print(f" -> Guardado: {save_path} (Pronto para Slicer)")

        # ------------------------------------------------------------------
        # VISUALIZAÇÃO (Usamos a versão 128 para ser rápido no Python)
        # ------------------------------------------------------------------
        print(" -> A abrir visualizador... (Verifica as cores do GT agora)")
        
        gt_resized = None
        if mask_data is not None:
            # Order 0 vital para não estragar labels
            gt_resized = robust_resize(mask_data, order=0)

        visualize_interactive(img_resized, pred_mask_128, gt_resized, case['id'])

if __name__ == "__main__":
    run_pipeline()