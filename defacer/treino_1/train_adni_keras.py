import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.optimizers import Adam
from scipy import ndimage

# Importa a nossa biblioteca
import model_adni_keras as model_lib

# --- CONFIGURAÇÃO ---
# Caminho do Cluster (Verifica se está correto)
BASE_DIR = "/home/andresousa615/defacer/ADNI_Lite_Cluster" 
NEW_MODEL_NAME = 'modelo_separado_nariz_boca.h5'
BATCH_SIZE = 1 
EPOCHS = 50
LEARNING_RATE = 1e-4

# --- GERADOR DE DADOS BLINDADO ---
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

    def robust_resize(self, volume, order=1):
        """
        Função auxiliar que garante que o volume sai EXATAMENTE com as dimensões self.dim
        """
        # 1. Remover 4ª dimensão se existir (ex: 256, 256, 160, 1) -> (256, 256, 160)
        if len(volume.shape) == 4:
            volume = volume[:, :, :, 0]

        # 2. Calcular fatores de Zoom
        factors = (
            self.dim[0] / volume.shape[0],
            self.dim[1] / volume.shape[1],
            self.dim[2] / volume.shape[2]
        )
        
        # 3. Aplicar Zoom
        # order=1 (linear) para imagens, order=0 (nearest) para máscaras
        vol_res = ndimage.zoom(volume, factors, order=order)

        # 4. ENCAIXE FORÇADO (Padding ou Crop)
        # Criamos uma caixa perfeita com o tamanho alvo
        final_vol = np.zeros(self.dim)
        
        # Descobrir as dimensões mínimas entre o resultado do zoom e o alvo
        # Isto evita erros se o zoom der 129 ou 85 pixeis
        d0 = min(self.dim[0], vol_res.shape[0])
        d1 = min(self.dim[1], vol_res.shape[1])
        d2 = min(self.dim[2], vol_res.shape[2])

        # Copiar o conteúdo para dentro da caixa (começando no 0,0,0)
        final_vol[:d0, :d1, :d2] = vol_res[:d0, :d1, :d2]

        return final_vol

    def __data_generation(self, list_ids_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes))

        for i, ID in enumerate(list_ids_temp):
            subj_path = os.path.join(self.base_dir, ID)
            raw_path = os.path.join(subj_path, "raw.nii.gz")
            
            # Tenta ler a máscara v4, senão usa a v3
            mask_path = os.path.join(subj_path, "training_masks", "mask_4_classes.nii.gz")
            if not os.path.exists(mask_path):
                 mask_path = os.path.join(subj_path, "training_masks", "mask_3_classes.nii.gz")

            # Carregar NIfTI
            img = nib.load(raw_path).get_fdata()
            msk = nib.load(mask_path).get_fdata()

            # --- AQUI ESTÁ A CORREÇÃO ---
            # Usamos a função robusta para garantir 128x128x128
            img = self.robust_resize(img, order=1)
            msk = self.robust_resize(msk, order=0)

            # Normalização
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

            # Limpeza de Labels (garantir que não há lixo acima de 4)
            msk = np.round(msk) # O zoom pode ter criado valores decimais na máscara
            msk[msk > 4] = 0 
            
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
        print("⚠️ AVISO: GPU NÃO DETETADA.")
    print("-" * 30)

    # 2. Dados
    all_patients = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    valid_patients = []
    
    # Validação rápida
    for p in all_patients:
        path_v4 = os.path.join(BASE_DIR, p, "training_masks", "mask_4_classes.nii.gz")
        path_v3 = os.path.join(BASE_DIR, p, "training_masks", "mask_3_classes.nii.gz")
        if os.path.exists(path_v4) or os.path.exists(path_v3):
            valid_patients.append(p)
            
    print(f"Pacientes Válidos: {len(valid_patients)}")
    if len(valid_patients) == 0: return

    train_ids, val_ids = train_test_split(valid_patients, test_size=0.2, random_state=42)
    
    # 3. Geradores
    gen_train = AdniDataGenerator(train_ids, BASE_DIR, batch_size=BATCH_SIZE)
    gen_val = AdniDataGenerator(val_ids, BASE_DIR, batch_size=BATCH_SIZE)

    # 4. Construir Modelo
    print("A construir modelo U-Net 3D (5 Classes)...")
    model = model_lib.get_unet_model(input_shape=(128, 128, 128, 1), n_classes=5)
    
    # 5. Compilar
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss=model_lib.dice_loss, 
                  metrics=['accuracy', model_lib.dice_score])
    
    # 6. Callbacks
    callbacks = [
        ModelCheckpoint(NEW_MODEL_NAME, monitor='val_dice_score', mode='max', save_best_only=True, verbose=1),
        CSVLogger('training_log.csv', append=True),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    ]

    # 7. Treinar
    print("A iniciar treino...")
    model.fit(
        gen_train,
        validation_data=gen_val,
        epochs=EPOCHS,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True,
        verbose=1
    )

if __name__ == "__main__":
    train()