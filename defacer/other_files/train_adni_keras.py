import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from scipy import ndimage


# Importa a tua arquitetura (não precisamos de alterar o ficheiro do modelo)
import model_adni_keras as model_lib

# --- CONFIGURAÇÃO ---
ADNI_DIR = "/home/andresousa615/defacer/ADNI_Lite_Cluster"  # Ajusta para o teu caminho do ADNI
IXI_DIR = "/home/andresousa615/defacer/IXI_Cluster" # Caminho dos exames IXI segmentados
NEW_MODEL_NAME = 'modelo_paper_replication.h5'
BATCH_SIZE = 1
EPOCHS = 60    
LEARNING_RATE = 1e-4

# --- FUNÇÕES DE AUGMENTATION (Mantidas do teu código) ---
def random_rotation(x, y, range_degrees=15): # Paper usa +/- 15 graus [cite: 114]
    angle = np.random.uniform(-range_degrees, range_degrees)
    axes = tuple(np.random.choice([0, 1, 2], 2, replace=False))
    x = ndimage.rotate(x, angle, axes=axes, reshape=False, order=1, mode='nearest')
    y = ndimage.rotate(y, angle, axes=axes, reshape=False, order=0, mode='nearest')
    return x, y

def random_shift(x, y, range_pixels=10): # Paper: shift 0 to 0.10 [cite: 114]
    shift_x = np.random.uniform(-range_pixels, range_pixels)
    shift_y = np.random.uniform(-range_pixels, range_pixels)
    shift_z = np.random.uniform(-range_pixels, range_pixels)
    x = ndimage.shift(x, (shift_x, shift_y, shift_z), order=1, mode='nearest')
    y = ndimage.shift(y, (shift_x, shift_y, shift_z), order=0, mode='nearest')
    return x, y

def random_flip(x, y):
    if np.random.random() > 0.5:
        axis = np.random.choice([0, 1, 2]) 
        x = np.flip(x, axis=axis)
        y = np.flip(y, axis=axis)
    return x, y

def random_zoom(x, y, zoom_range=(0.9, 1.1)): # Paper: 0.90 to 1.10 [cite: 114]
    zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
    x = ndimage.zoom(x, zoom_factor, order=1)
    y = ndimage.zoom(y, zoom_factor, order=0)
    # Crop/Pad para manter 128x128x128
    return crop_or_pad(x, y)

def crop_or_pad(x, y, target_shape=(128, 128, 128)):
    # Helper para garantir dimensão fixa após zoom
    final_x = np.zeros(target_shape)
    final_y = np.zeros(target_shape)
    
    dx = (x.shape[0] - target_shape[0]) // 2
    dy = (x.shape[1] - target_shape[1]) // 2
    dz = (x.shape[2] - target_shape[2]) // 2
    
    src_x = slice(max(0, dx), max(0, dx) + min(x.shape[0], target_shape[0]))
    src_y = slice(max(0, dy), max(0, dy) + min(x.shape[1], target_shape[1]))
    src_z = slice(max(0, dz), max(0, dz) + min(x.shape[2], target_shape[2]))
    
    dst_x = slice(max(0, -dx), max(0, -dx) + min(x.shape[0], target_shape[0]))
    dst_y = slice(max(0, -dy), max(0, -dy) + min(x.shape[1], target_shape[1]))
    dst_z = slice(max(0, -dz), max(0, -dz) + min(x.shape[2], target_shape[2]))

    final_x[dst_x, dst_y, dst_z] = x[src_x, src_y, src_z]
    final_y[dst_x, dst_y, dst_z] = y[src_x, src_y, src_z]
    
    return final_x, final_y

# --- GERADOR DE DADOS ---
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_ids, base_dir, batch_size=1, dim=(128,128,128), n_classes=5, shuffle=True, augment=False, augmentation_factor=1):
        self.list_ids = list_ids
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.augmentation_factor = augmentation_factor # Fator para replicar o volume do paper
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle: np.random.shuffle(self.indexes)

    def __len__(self):
        # Multiplicamos pelo augmentation_factor para simular "576 images per epoch"
        # Se augmentation_factor = 4, a época será 4x mais longa.
        return int(np.floor((len(self.list_ids) * self.augmentation_factor) / self.batch_size))

    def __getitem__(self, index):
        # Ajuste do índice para fazer "wrap around" na lista original
        real_index = index % (len(self.list_ids) // self.batch_size)
        
        indexes = self.indexes[real_index*self.batch_size:(real_index+1)*self.batch_size]
        list_ids_temp = [self.list_ids[k] for k in indexes]
        return self.__data_generation(list_ids_temp)

    def robust_resize(self, volume, order=1):
        if len(volume.shape) == 4: volume = volume[:, :, :, 0]
        # Normalização Min-Max para resize limpo
        min_v, max_v = np.min(volume), np.max(volume)
        volume = (volume - min_v) / (max_v - min_v + 1e-8)
        
        factors = (self.dim[0]/volume.shape[0], self.dim[1]/volume.shape[1], self.dim[2]/volume.shape[2])
        vol_res = ndimage.zoom(volume, factors, order=order)
        return vol_res

    def remap_labels(self, mask):
        # Mapeia etiquetas (assumindo output do teu script mask_divider/reprocess)
        # 0=Fundo, 1=Face(Resto), 2=Orelhas, 3=Boca, 4=Nariz, 5=Olhos
        # U-Net espera: 0=Fundo, 1=Face/Orelhas?, 2=Boca, 3=Nariz, 4=Olhos (Ajusta confome a tua rede)
        # Assumindo a tua configuração atual:
        new_mask = np.copy(mask)
        # AQUI É CRÍTICO: Garante que os valores batem certo com o que a tua U-Net espera (5 classes)
        # Se a tua U-Net tem 5 canais, os valores devem ser 0, 1, 2, 3, 4.
        # No teu mask_4_classes tens: 0, 1(Face), 2(Orelhas), 3(Boca), 4(Nariz), 5(Olhos)?
        # Se o output layer tem 5 canais, tens de mapear para 0-4.
        
        # Vamos assumir: 0=Background, 1=Orelhas, 2=Boca, 3=Nariz, 4=Olhos
        final_mask = np.zeros_like(mask)
        final_mask[mask == 2] = 1 # Orelhas
        final_mask[mask == 3] = 2 # Boca
        final_mask[mask == 4] = 3 # Nariz
        final_mask[mask == 5] = 4 # Olhos
        # Nota: Ignoramos a "Face Rest" (1) se o objetivo for só os órgãos, ou mapeamos para background.
        
        return final_mask

    def __data_generation(self, list_ids_temp):
        X = np.empty((self.batch_size, *self.dim, 1))
        y = np.empty((self.batch_size, *self.dim, self.n_classes))

        for i, ID in enumerate(list_ids_temp):
            subj_path = os.path.join(self.base_dir, ID)
            try:
                # Load raw
                raw_path = os.path.join(subj_path, "raw.nii.gz")
                if not os.path.exists(raw_path): # Fallback
                     cand = [f for f in os.listdir(subj_path) if f.endswith(".nii.gz") and "mask" not in f][0]
                     raw_path = os.path.join(subj_path, cand)
                
                img = nib.load(raw_path).get_fdata()
                
                # Load mask
                mask_path = os.path.join(subj_path, "training_masks", "mask_4_classes.nii.gz")
                msk = nib.load(mask_path).get_fdata()
            except Exception as e:
                print(f"Erro {ID}: {e}")
                continue

            # Resize
            img = self.robust_resize(img, order=1)
            msk = self.robust_resize(msk, order=0)
            
            # Remap para 0-4 (5 classes)
            msk = np.round(msk)
            msk = self.remap_labels(msk)

            # Augmentation
            if self.augment:
                # Randomly apply augmentations
                if np.random.random() > 0.5: img, msk = random_rotation(img, msk)
                if np.random.random() > 0.5: img, msk = random_shift(img, msk)
                if np.random.random() > 0.5: img, msk = random_zoom(img, msk)
                if np.random.random() > 0.5: img, msk = random_flip(img, msk)

            # Normalize Intensity
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
            
            msk_onehot = to_categorical(msk, num_classes=self.n_classes)

            X[i, ] = np.expand_dims(img, axis=-1)
            y[i, ] = msk_onehot

        return X, y

# --- LOSS DO PAPER ---
# O paper usa: 1 - Dice + 0.1 * CrossEntropy 
def paper_loss(y_true, y_pred):
    dice = model_lib.dice_score(y_true, y_pred)
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return (1 - dice) + (0.1 * cce)

# --- EXECUÇÃO PRINCIPAL ---
def run_replication():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: 
        try: tf.config.experimental.set_memory_growth(gpus[0], True); 
        except: pass

    # 1. PREPARAR DADOS ADNI (240 Exames)
    print("--- 1. A carregar ADNI ---")
    all_adni = [f for f in os.listdir(ADNI_DIR) if os.path.isdir(os.path.join(ADNI_DIR, f))]
    valid_adni = [f for f in all_adni if os.path.exists(os.path.join(ADNI_DIR, f, "training_masks", "mask_4_classes.nii.gz"))]
    
    # Filtrar IDs únicos para evitar fuga de dados (data leakage)
    patient_ids = [f.split('__')[0] for f in valid_adni] # Ajusta o split se necessário
    unique_patients = list(set(patient_ids))
    
    # --- DIVISÃO PAPER: 144 / 48 / 48 ---
    # Primeiro separamos 20% para Teste (48)
    train_val_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)
    
    # Dos restantes (192), separamos 25% para Validação (48) -> Sobram 144 para Treino
    # (48 é 25% de 192)
    train_patients, val_patients = train_test_split(train_val_patients, test_size=0.25, random_state=42)
    
    # Mapear de volta para as pastas
    train_ids = [f for f in valid_adni if f.split('__')[0] in train_patients]
    val_ids = [f for f in valid_adni if f.split('__')[0] in val_patients]
    test_ids = [f for f in valid_adni if f.split('__')[0] in test_patients]
    
    # --- VERIFICAÇÃO DE SEGURANÇA ANTI-LEAKAGE ---
    set_train = set([x.split('__')[0] for x in train_ids])
    set_val = set([x.split('__')[0] for x in val_ids])
    set_test = set([x.split('__')[0] for x in test_ids])
    
    leak_tv = set_train.intersection(set_val)
    leak_tt = set_train.intersection(set_test)
    leak_vt = set_val.intersection(set_test)
    
    if len(leak_tv) > 0 or len(leak_tt) > 0 or len(leak_vt) > 0:
        print("🚨 ERRO CRÍTICO: FUGA DE DADOS DETETADA!")
        print(f"Pacientes em duplicado: {leak_tv | leak_tt | leak_vt}")
        raise ValueError("Abortar treino: O mesmo paciente está em sets diferentes.")
    else:
        print("✅ SEGURANÇA: Nenhum paciente se repete entre Treino, Validação e Teste.")
    # ==============================================================================

    print(f"SPLIT (Paper Replication):")
    print(f" -> Treino: {len(train_ids)} (Meta: ~144)")
    print(f" -> Valid : {len(val_ids)} (Meta: ~48)")
    print(f" -> Teste : {len(test_ids)} (Meta: ~48)")
    
    # 2. CONFIGURAR GERADORES
    # augmentation_factor=4 garante 576 samples/epoch (144 * 4) [cite: 114]
    gen_train = DataGenerator(train_ids, ADNI_DIR, batch_size=BATCH_SIZE, augment=True, augmentation_factor=4)
    gen_val = DataGenerator(val_ids, ADNI_DIR, batch_size=BATCH_SIZE, augment=False, augmentation_factor=1)

    # 3. MODELO
    print("--- 2. A Compilar Modelo ---")
    model = model_lib.get_unet_model(input_shape=(128, 128, 128, 1), n_classes=5)
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss=paper_loss, # Usamos a loss exata do paper
                  metrics=['accuracy', model_lib.dice_score])
    
    callbacks = [
        ModelCheckpoint(NEW_MODEL_NAME, monitor='val_dice_score', mode='max', save_best_only=True, verbose=1),
        CSVLogger('training_log_paper_rep.csv', append=True),
        EarlyStopping(monitor='val_dice_score', patience=10, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    # 4. TREINO
    print("--- 3. A Treinar (Replication) ---")
    model.fit(
        gen_train, 
        validation_data=gen_val, 
        epochs=EPOCHS, 
        callbacks=callbacks, 
        verbose=1,
        workers=2,
        max_queue_size=2,
        use_multiprocessing=True 
    )

    # 5. AVALIAÇÃO FINAL
    print("\n--- 4. Avaliação Final ---")
    # Carregar o melhor modelo
    best_model = tf.keras.models.load_model(NEW_MODEL_NAME, custom_objects={
        'dice_loss': model_lib.dice_loss, 
        'dice_score': model_lib.dice_score, 
        'paper_loss': paper_loss,
        'InstanceNormalization': model_lib.InstanceNormalization
    })
    
    # Avaliar ADNI Test Set (Internal)
    print(" -> A avaliar Test Set Interno (ADNI)...")
    gen_test_adni = DataGenerator(test_ids, ADNI_DIR, batch_size=1, augment=False)
    scores_adni = best_model.evaluate(gen_test_adni, verbose=1)
    print(f"ADNI Test Dice: {scores_adni[2]:.4f}")
    
    # Avaliar IXI Test Set (External)
    print(" -> A avaliar Test Set Externo (IXI)...")
    valid_ixi = [f for f in os.listdir(IXI_DIR) if os.path.exists(os.path.join(IXI_DIR, f, "training_masks", "mask_4_classes.nii.gz"))]
    
    # Selecionar 100 exames IXI se houver mais, para replicar o N=100 do paper [cite: 99]
    if len(valid_ixi) > 100: valid_ixi = valid_ixi[:100]
    
    if len(valid_ixi) > 0:
        gen_test_ixi = DataGenerator(valid_ixi, IXI_DIR, batch_size=1, augment=False)
        scores_ixi = best_model.evaluate(gen_test_ixi, verbose=1)
        print(f"IXI External Test Dice: {scores_ixi[2]:.4f}")
    else:
        print("AVISO: Nenhum exame IXI válido encontrado para teste.")

if __name__ == "__main__":
    run_replication()
