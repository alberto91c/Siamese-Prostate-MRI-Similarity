
# coding: utf-8


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras import backend as K
from keras.losses import mean_squared_error
import numpy as np
import math
from keras.callbacks import TensorBoard, ModelCheckpoint


# load ROI: SELECT TYPE OF IMAGE
data=np.load("./Data/roihbvSmall.npy")
# scale image
data=data/data.max()

# corresponding index for train, validation, and test image
train=np.load("./PatternSelection/ThirdRun/index_train.npy")
validation=np.load("./PatternSelection/ThirdRun/index_val.npy")
test=np.load("./PatternSelection/ThirdRun/index_test.npy")

#eventually add weighting mask: does not give relevant improvement for autoencoder
'''
vec=[]
for i in range(0,40):
    vec.append(math.exp(-((i-20)**2)/(2*(5**2))))

xy=np.zeros((40,40,3))
for i in range(0,40):
    for j in range(0,40):
        xy[i][j][0]=vec[i]*vec[j]
        xy[i][j][1]=vec[i]*vec[j]
        xy[i][j][2]=vec[i]*vec[j]

for i in range(0,data.shape[0]):
    data[i,:,:,:]=data[i,:,:,:]*xy
'''

# split train, validation and test based on the corresponding index
x_train=data[train]
x_val=data[validation]
x_test=data[test]

# define the model
input_img = Input(shape=(data.shape[1], data.shape[2], data.shape[3]))  # adapt this if using `channels_first` image data format

# encoder
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same',name="enc")(x)

# decoder
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(x)

autoencoder = Model(input_img, decoded)

# set checkpoint, optimizer, loss and train the model
autoencoder.compile(optimizer='adam', loss='mse')
checkpoint = ModelCheckpoint('WAutoencoderHBV.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
autoencoder.fit(x_train, x_train,
                epochs=2000,
                batch_size=64,
                shuffle=True,
                validation_data=(x_val, x_val), callbacks=[checkpoint])

# reload best model
del autoencoder
model=load_model("./WAutoencoderHBV.h5")

# extract feature from the encoder
layer_name = 'enc'
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
train_enc = intermediate_layer_model.predict(x_train)
val_enc = intermediate_layer_model.predict(x_val)
test_enc = intermediate_layer_model.predict(x_test)


# save the representation
np.save("train_WencHBV3.npy", train_enc)
np.save("val_WencHBV3.npy", val_enc)
np.save("test_WencHBV3.npy", test_enc)
