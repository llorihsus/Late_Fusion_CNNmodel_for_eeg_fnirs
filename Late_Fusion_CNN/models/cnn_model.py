import tensorflow as tf
from keras.models import Model


from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate, Lambda, Dense
from tensorflow.keras import Input


from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Normalization

#EF-Net
def eeg_fnirs_cnn():
    tf.keras.backend.clear_session()

    inputE = Input(shape=(1000, 30,1))
    inputF = Input(shape=(50, 72,1))

    # denoising eeg
    e = Conv2D(filters=8, kernel_size=(15,1), padding='same', activation='relu')(inputE)
    e = BatchNormalization()(e)
    e = Conv2D(filters=8, kernel_size=(1,3), padding='same', activation='relu')(e)
    e = BatchNormalization()(e)

    e = Conv2D(filters=32, kernel_size=(7,1), activation='relu')(e)   
    e = Conv2D(filters=32, kernel_size=(7,1), activation='relu')(e)
    e = Conv2D(filters=32, kernel_size=(7,1), activation='relu')(e)
    e= MaxPooling2D(pool_size=(7,1))(e)   
    e= Dropout(0.5)(e)
    e= BatchNormalization()(e)

    e =Conv2D(filters=64, kernel_size=(4,4),activation='relu')(e)             
    e =Conv2D(filters=64, kernel_size=(4,4),activation='relu')(e)
    e =Conv2D(filters=64, kernel_size=(4,4),activation='relu')(e)
    e= MaxPooling2D(pool_size=(4,4))(e)
    e= Dropout(0.5)(e)
    e= BatchNormalization()(e)
    e= Flatten()(e)

    e=Dense(256,activation="relu")(e)
    e= Dropout(0.5)(e)
    e=Dense(128,activation="relu")(e)
    e = Model(inputs=inputE, outputs=e)

    # denoising fnirs
    f = Conv2D(filters=8, kernel_size=(7,1), padding='same', activation='relu')(inputF)
    f = BatchNormalization()(f)
    f = Conv2D(filters=8, kernel_size=(1,5), padding='same', activation='relu')(f)
    f = BatchNormalization()(f)


    f=Conv2D(filters=32, kernel_size=(4,1))(f)    
    f=Conv2D(filters=32, kernel_size=(4,1))(f)
    f= MaxPooling2D(pool_size=(4,1))(f)  
    f= Dropout(0.5)(f)
    f= BatchNormalization()(f)
    f=Conv2D(filters=64, kernel_size=(2,2))(f)
    f=Conv2D(filters=64, kernel_size=(2,2))(f)
    f= MaxPooling2D(pool_size=(2,2))(f)
    f= Dropout(0.5)(f)
    f= BatchNormalization()(f)
    f= Flatten()(f)
    f=Dense(128,activation="relu")(f)
    f = Model(inputs=inputF, outputs=f)

    combined = concatenate([e.output, f.output])
    #MLP
    z = Dense(256, activation="relu")(combined)
    z = Dropout(0.5)(z)
    z = Lambda(lambda x: tf.math.l2_normalize(x, axis=1, epsilon=5e-4))(z)
    z = Dense(64, activation="relu")(z)
    z = Dense(3, activation="softmax")(z)  # ← new output layer

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[e.input, f.input], outputs=z)
    return model
