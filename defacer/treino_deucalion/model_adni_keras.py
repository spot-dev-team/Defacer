import numpy as np
from scipy import ndimage
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, BatchNormalization, Activation, SpatialDropout3D
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints

# --- MÉTICAS E LOSS ---
def dice_score(y_true, y_pred):
    smooth = 1.
    label_length = y_pred.shape[-1]
    loss = 0    
    for num_labels in range(label_length):
        y_true_f = K.flatten(y_true[..., num_labels])
        y_pred_f = K.flatten(y_pred[..., num_labels])
        intersection = K.sum(y_true_f * y_pred_f)
        loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return loss/label_length 

def dice_loss(y_true, y_pred):
    return 1-dice_score(y_true, y_pred) + 0.01*K.categorical_crossentropy(y_true, y_pred)

# --- INSTANCE NORMALIZATION (Original) ---
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
        ndim = len(input_shape)
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

# --- ARQUITETURA U-NET 3D (NOVO: Para treinar do zero) ---
def get_unet_model(input_shape=(128, 128, 128, 1), n_classes=5):
    inputs = Input(input_shape)
    
    # Encoder
    # Bloco 1
    c1 = Conv3D(16, (3, 3, 3), padding='same')(inputs)
    c1 = InstanceNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv3D(32, (3, 3, 3), padding='same')(c1)
    c1 = InstanceNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling3D(pool_size=(2, 2, 2))(c1)
    
    # Bloco 2
    c2 = Conv3D(32, (3, 3, 3), padding='same')(p1)
    c2 = InstanceNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv3D(64, (3, 3, 3), padding='same')(c2)
    c2 = InstanceNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling3D(pool_size=(2, 2, 2))(c2)
    
    # Bloco 3
    c3 = Conv3D(64, (3, 3, 3), padding='same')(p2)
    c3 = InstanceNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv3D(128, (3, 3, 3), padding='same')(c3)
    c3 = InstanceNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling3D(pool_size=(2, 2, 2))(c3)
    
    # Bottleneck
    c4 = Conv3D(128, (3, 3, 3), padding='same')(p3)
    c4 = InstanceNormalization()(c4)
    c4 = Activation('relu')(c4)
    c4 = Conv3D(256, (3, 3, 3), padding='same')(c4)
    c4 = InstanceNormalization()(c4)
    c4 = Activation('relu')(c4)
    
    # Decoder
    # Up 1
    u5 = UpSampling3D(size=(2, 2, 2))(c4)
    u5 = concatenate([u5, c3], axis=-1)
    c5 = Conv3D(128, (3, 3, 3), padding='same')(u5)
    c5 = InstanceNormalization()(c5)
    c5 = Activation('relu')(c5)
    c5 = Conv3D(128, (3, 3, 3), padding='same')(c5)
    c5 = InstanceNormalization()(c5)
    c5 = Activation('relu')(c5)
    
    # Up 2
    u6 = UpSampling3D(size=(2, 2, 2))(c5)
    u6 = concatenate([u6, c2], axis=-1)
    c6 = Conv3D(64, (3, 3, 3), padding='same')(u6)
    c6 = InstanceNormalization()(c6)
    c6 = Activation('relu')(c6)
    c6 = Conv3D(64, (3, 3, 3), padding='same')(c6)
    c6 = InstanceNormalization()(c6)
    c6 = Activation('relu')(c6)

    # Up 3
    u7 = UpSampling3D(size=(2, 2, 2))(c6)
    u7 = concatenate([u7, c1], axis=-1)
    c7 = Conv3D(32, (3, 3, 3), padding='same')(u7)
    c7 = InstanceNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Conv3D(32, (3, 3, 3), padding='same')(c7)
    c7 = InstanceNormalization()(c7)
    c7 = Activation('relu')(c7)
    
    # Output Layer (Softmax para 5 classes)
    outputs = Conv3D(n_classes, (1, 1, 1), activation='softmax')(c7)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Função Auxiliar Resize
def resize_volume(data, img_dep=128, img_cols=128, img_rows=128):
    resize_factor = (img_dep/data.shape[0], img_cols/data.shape[1], img_rows/data.shape[2])
    data = ndimage.zoom(data, resize_factor, order=0, mode='nearest')
    return data


# Função Auxiliar Remap Labels (Centralizada)
def remap_labels(mask):
    """Mapeia os labels originais (2,3,4,5) para o formato do treino (1,2,3,4)."""
    import numpy as np
    final_mask = np.zeros_like(mask)
    final_mask[mask == 2] = 1 # Orelhas
    final_mask[mask == 3] = 2 # Boca
    final_mask[mask == 4] = 3 # Nariz
    final_mask[mask == 5] = 4 # Olhos
    return final_mask