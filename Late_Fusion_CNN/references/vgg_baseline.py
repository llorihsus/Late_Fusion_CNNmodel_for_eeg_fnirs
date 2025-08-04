import tensorflow as tf
from keras.models import Model


from tensorflow.keras.layers import Flatten, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate, Lambda, Dense
from tensorflow.keras import Input


from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Normalization


#Baseline
def eeg_fnirs_vgg(eeg_train_data,fnirs_train_data,resnet):
    inputE = Input(shape=(200, 32,3))
    inputF = Input(shape=(32, 72, 3))

    normalization_layerE = Normalization()
    normalization_layerF = Normalization()
    normalization_layerE.adapt(eeg_train_data)   
    normalization_layerF.adapt(fnirs_train_data)     
    norm_layerE = normalization_layerE(inputE)
    norm_layerF = normalization_layerF(inputF)

    # eeg denoising layer
    e = Conv2D(filters=8, kernel_size=(15,1), padding='same', activation='relu')(norm_layerE)
    e = BatchNormalization()(e)
    e = Conv2D(filters=8, kernel_size=(1,3), padding='same', activation='relu')(e)
    e = BatchNormalization()(e)

    if resnet:
      baseline_modele= tf.keras.applications.ResNet50(include_top=False,input_shape=(200, 32,3), classes=2, weights=None)
    else:
      baseline_modele= tf.keras.applications.VGG19(include_top=False,input_shape=(200, 32,3),weights=None) 

    baseline_modelf =vgg16.VGG16(weights=None, include_top=False, input_shape=(32, 72, 3))

    # the first branch operates on the first input: EEG
    e = (baseline_modele)(norm_layerE)
    e = Flatten()(e)
    e = Dense(128, activation='relu')(e)
    e = Model(inputs=inputE, outputs=e)

    # fnirs denoising layer
    f = Conv2D(filters=8, kernel_size=(7,1), padding='same', activation='relu')(norm_layerF)
    f = BatchNormalization()(f)
    f = Conv2D(filters=8, kernel_size=(1,5), padding='same', activation='relu')(f)
    f = BatchNormalization()(f)

    # the second branch operates on the second input: FNIRS
    f = (baseline_modelf)(norm_layerF)
    f = Flatten()(f)
    f = Dense(128, activation='relu')(f)
    f = Model(inputs=inputF, outputs=f)
    # combine the output of the two branches
    combined = concatenate([e.output, f.output])

    #MLP
    z = Dense(256, activation="relu")(combined)
    z = Dense(64, activation="relu")(z)
    z = Dense(3, activation="softmax")(z)  # ← new output layer

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[e.input, f.input], outputs=z)
    return model
