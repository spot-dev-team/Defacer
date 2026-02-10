import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Importa a nossa biblioteca
import model_adni_keras as model_lib

# --- CONFIGURAÇÃO ---
BASE_DIR = r"D:\Tese_BDs\Defacer\ADNI_Structured"
NEW_MODEL_NAME = 'modelo_treinado_do_zero.h5'
BATCH_SIZE = 1 
EPOCHS = 50
LEARNING_RATE = 1e-4

# --- GERADOR DE DADOS ---
class AdniDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_ids, base_dir, batch_size=1, dim=(128,128,128), n_channels=1, n_classes=5, shuffle=True):
        self.list_ids = list_ids
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
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

    def __data_generation(self, list_ids_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes))

        for i, ID in enumerate(list_ids_temp):
            subj_path = os.path.join(self.base_dir, ID)
            raw_path = os.path.join(subj_path, "raw.nii.gz")
            # Usar a máscara de 3 classes (fundida)
            mask_path = os.path.join(subj_path, "training_masks", "mask_3_classes.nii.gz")
            
            img = nib.load(raw_path).get_fdata()
            msk = nib.load(mask_path).get_fdata()

            img = model_lib.resize_volume(img)
            msk = model_lib.resize_volume(msk)

            # Normalização Min-Max
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

            # Mapeamento 3 Classes -> 5 Classes Sequenciais
            # 0=Bg, 1=Face, 2=Ears, 3=LowerFace, 5=Eyes ---> 5 vira 4
            msk[msk == 5] = 4
            
            msk_onehot = to_categorical(msk, num_classes=self.n_classes)

            X[i, ] = np.expand_dims(img, axis=-1)
            y[i, ] = msk_onehot

        return X, y

# --- SCRIPT PRINCIPAL ---
def train():
    # 1. Configuração GPU
    print("-" * 30)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU DETETADA: {gpus[0]}")
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except: pass
    else:
        print("❌ AVISO: GPU NÃO DETETADA. O treino será lento.")
        print("   Erro provável: Faltam DLLs do CUDA (cudnn64_8.dll, etc).")
    print("-" * 30)

    # 2. Dados
    all_patients = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    valid_patients = []
    for p in all_patients:
        if os.path.exists(os.path.join(BASE_DIR, p, "training_masks", "mask_3_classes.nii.gz")):
            valid_patients.append(p)
            
    print(f"Pacientes Válidos: {len(valid_patients)}")
    if len(valid_patients) == 0: return

    train_ids, val_ids = train_test_split(valid_patients, test_size=0.2, random_state=42)
    
    gen_train = AdniDataGenerator(train_ids, BASE_DIR, batch_size=BATCH_SIZE)
    gen_val = AdniDataGenerator(val_ids, BASE_DIR, batch_size=BATCH_SIZE)

    # 3. Construir Modelo do Zero (AQUI ESTÁ A MUDANÇA)
    print("A construir modelo U-Net 3D do zero...")
    model = model_lib.get_unet_model(input_shape=(128, 128, 128, 1), n_classes=5)
    
    # 4. Compilar
    # Usamos Dice Loss como no original
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss=model_lib.dice_loss, 
                  metrics=['accuracy', model_lib.dice_score])
    
    model.summary()

    # 5. Callbacks
    callbacks = [
        ModelCheckpoint(NEW_MODEL_NAME, monitor='val_dice_score', mode='max', save_best_only=True, verbose=1),
        CSVLogger('training_log.csv', append=True),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    ]

    # 6. Treinar
    print("A iniciar treino...")
    model.fit(
        gen_train,
        validation_data=gen_val,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

if __name__ == "__main__":
    train()