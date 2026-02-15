import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.optimizers import Adam
from scipy import ndimage

# Importa a tua arquitetura
import model_adni_keras as model_lib

# --- CONFIGURAÇÃO (ALTERADA PARA O BASELINE) ---
BASE_DIR = "/home/andresousa615/defacer/ADNI_Lite_Cluster" 
NEW_MODEL_NAME = 'modelo_baseline_no_aug.h5'  # <--- MUDANÇA 1: Nome novo
BATCH_SIZE = 1 
EPOCHS = 60
LEARNING_RATE = 1e-4

# --- FUNÇÕES DE AUGMENTATION (Mantemos aqui mas não serão chamadas) ---
def random_rotation(x, y, range_degrees=10):
    angle = np.random.uniform(-range_degrees, range_degrees)
    axes = tuple(np.random.choice([0, 1, 2], 2, replace=False))
    x = ndimage.rotate(x, angle, axes=axes, reshape=False, order=1, mode='nearest')
    y = ndimage.rotate(y, angle, axes=axes, reshape=False, order=0, mode='nearest')
    return x, y

def random_shift(x, y, range_pixels=10):
    shift_x = np.random.uniform(-range_pixels, range_pixels)
    shift_y = np.random.uniform(-range_pixels, range_pixels)
    shift_z = np.random.uniform(-range_pixels, range_pixels)
    x = ndimage.shift(x, (shift_x, shift_y, shift_z), order=1, mode='nearest')
    y = ndimage.shift(y, (shift_x, shift_y, shift_z), order=0, mode='nearest')
    return x, y

def random_flip(x, y):
    if np.random.random() > 0.5:
        axis = np.random.choice([0, 1]) 
        x = np.flip(x, axis=axis)
        y = np.flip(y, axis=axis)
    return x, y

def random_zoom(x, y, zoom_range=(0.85, 1.15)):
    zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
    x = ndimage.zoom(x, zoom_factor, order=1)
    y = ndimage.zoom(y, zoom_factor, order=0)
    
    final_x = np.zeros((128, 128, 128))
    final_y = np.zeros((128, 128, 128))
    
    dx = (x.shape[0] - 128) // 2
    dy = (x.shape[1] - 128) // 2
    dz = (x.shape[2] - 128) // 2
    
    src_x_start = max(0, dx); src_y_start = max(0, dy); src_z_start = max(0, dz)
    dst_x_start = max(0, -dx); dst_y_start = max(0, -dy); dst_z_start = max(0, -dz)
    
    copy_x = min(x.shape[0] - src_x_start, 128 - dst_x_start)
    copy_y = min(x.shape[1] - src_y_start, 128 - dst_y_start)
    copy_z = min(x.shape[2] - src_z_start, 128 - dst_z_start)

    final_x[dst_x_start:dst_x_start+copy_x, dst_y_start:dst_y_start+copy_y, dst_z_start:dst_z_start+copy_z] = \
        x[src_x_start:src_x_start+copy_x, src_y_start:src_y_start+copy_y, src_z_start:src_z_start+copy_z]
    final_y[dst_x_start:dst_x_start+copy_x, dst_y_start:dst_y_start+copy_y, dst_z_start:dst_z_start+copy_z] = \
        y[src_x_start:src_x_start+copy_x, src_y_start:src_y_start+copy_y, src_z_start:src_z_start+copy_z]
        
    return final_x, final_y

# --- GERADOR DE DADOS ---
class AdniDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_ids, base_dir, batch_size=1, dim=(128,128,128), n_channels=1, n_classes=5, shuffle=True, augment=False):
        self.list_ids = list_ids
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment 
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle: np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_ids_temp = [self.list_ids[k] for k in indexes]
        return self.__data_generation(list_ids_temp)

    def robust_resize(self, volume, order=1):
        if len(volume.shape) == 4: volume = volume[:, :, :, 0]
        factors = (self.dim[0]/volume.shape[0], self.dim[1]/volume.shape[1], self.dim[2]/volume.shape[2])
        vol_res = ndimage.zoom(volume, factors, order=order)
        final_vol = np.zeros(self.dim)
        d0, d1, d2 = min(self.dim[0], vol_res.shape[0]), min(self.dim[1], vol_res.shape[1]), min(self.dim[2], vol_res.shape[2])
        final_vol[:d0, :d1, :d2] = vol_res[:d0, :d1, :d2]
        return final_vol

    def remap_labels(self, mask):
        """
        Mapeia os segmentos do Slicer (com lixo) para as classes da U-Net.
        Entrada (Slicer): 0=Fundo, 1=Lixo, 2=Orelhas, 3=Boca, 4=Nariz, 5=Olhos
        Saída (U-Net):    0=Fundo, 1=Orelhas, 2=Boca, 3=Nariz, 4=Olhos
        """
        new_mask = np.zeros_like(mask)
        new_mask[mask == 2] = 1 # Orelhas
        new_mask[mask == 3] = 2 # Boca
        new_mask[mask == 4] = 3 # Nariz
        new_mask[mask == 5] = 4 # Olhos
        return new_mask

    def __data_generation(self, list_ids_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes))

        for i, ID in enumerate(list_ids_temp):
            subj_path = os.path.join(self.base_dir, ID)
            try:
                img = nib.load(os.path.join(subj_path, "raw.nii.gz")).get_fdata()
                mask_path = os.path.join(subj_path, "training_masks", "mask_4_classes.nii.gz")
                
                if not os.path.exists(mask_path):
                     continue
                     
                msk = nib.load(mask_path).get_fdata()
            except Exception as e:
                print(f"Erro ao ler {ID}: {e}")
                continue

            # 1. Resize Inicial
            img = self.robust_resize(img, order=1)
            msk = self.robust_resize(msk, order=0)

            # 2. REMAPEAMENTO (CRÍTICO - Mantemos isto!)
            msk = np.round(msk)
            msk = self.remap_labels(msk)

            # 3. DATA AUGMENTATION (Desativado se self.augment=False)
            if self.augment:
                if np.random.random() > 0.5: img, msk = random_rotation(img, msk)
                if np.random.random() > 0.5: img, msk = random_shift(img, msk)
                if np.random.random() > 0.5: img, msk = random_flip(img, msk)
                if np.random.random() > 0.5: img, msk = random_zoom(img, msk)

            # 4. Normalização
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
            
            msk = np.round(msk)
            msk_onehot = to_categorical(msk, num_classes=self.n_classes)

            X[i, ] = np.expand_dims(img, axis=-1)
            y[i, ] = msk_onehot

        return X, y

# --- LOOP DE TREINO ---
def train():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: 
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except: pass

    all_folders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    
    valid_folders = []
    for f in all_folders:
        path_v4 = os.path.join(BASE_DIR, f, "training_masks", "mask_4_classes.nii.gz")
        if os.path.exists(path_v4):
            valid_folders.append(f)
            
    print(f"Total de Exames Válidos: {len(valid_folders)}")
    if len(valid_folders) == 0: 
        print("Nenhum dado encontrado!")
        return

    # Divisão por Paciente
    try:
        patient_ids = [f.split('__')[0] for f in valid_folders]
    except:
        patient_ids = valid_folders

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(valid_folders, groups=patient_ids))
    
    train_ids = [valid_folders[i] for i in train_idx]
    val_ids = [valid_folders[i] for i in val_idx]
    
    # Segurança
    intersection = set([f.split('__')[0] for f in train_ids]).intersection(set([f.split('__')[0] for f in val_ids]))
    if len(intersection) > 0:
        print(f"🚨 ERRO CRÍTICO: FUGA DE DADOS! {intersection}")
        return

    print(f"Treino: {len(train_ids)} | Validação: {len(val_ids)}")

    # --- MUDANÇA 2: augment=False no Treino ---
    gen_train = AdniDataGenerator(train_ids, BASE_DIR, batch_size=BATCH_SIZE, augment=False)
    gen_val = AdniDataGenerator(val_ids, BASE_DIR, batch_size=BATCH_SIZE, augment=False)

    model = model_lib.get_unet_model(input_shape=(128, 128, 128, 1), n_classes=5)
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss=model_lib.dice_loss, 
                  metrics=['accuracy', model_lib.dice_score])
    
    callbacks = [
        ModelCheckpoint(NEW_MODEL_NAME, monitor='val_dice_score', mode='max', save_best_only=True, verbose=1),
        CSVLogger('training_log_no_aug.csv', append=True), # <--- MUDANÇA 3: Log novo
        EarlyStopping(monitor='val_loss', patience=15, verbose=1)
    ]

    model.fit(gen_train, validation_data=gen_val, epochs=EPOCHS, callbacks=callbacks, workers=4, use_multiprocessing=True, verbose=1)

if __name__ == "__main__":
    train()