#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D, BatchNormalization, Dropout, concatenate
from keras.models import Model, Sequential, load_model
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
import math

# set batch size and epochs
batch_size=64
epochs=30

# load couple index and target of each couple
target=np.load("./Couple/FirstRun/Alltarget.npy")
couple=np.load("./Couple/FirstRun/Allcouple.npy")
target_val=np.load("./Couple/FirstRun/AlltargetVal.npy")
couple_val=np.load("./Couple/FirstRun/Allcouple_Val.npy")

# load data and scale
hbv=np.load("./Data/roihbvSmall.npy")
t2w=np.load("./Data/roit2w.npy")[:,20:60,20:60,:]
sag=np.load("./Data/roit2wSag.npy")[:,20:60,20:60,:]
t2w=t2w/np.max(t2w)
hbv=hbv/np.max(hbv)
sag=sag/np.max(sag)

# apply mask to weight more the center of the image: it give an improvement
vec=[]
for i in range(0,40):
    vec.append(math.exp(-((i-20)**2)/(2*(5**2))))

xy=np.zeros((40,40,3))
for i in range(0,40):
    for j in range(0,40):
        xy[i][j][0]=vec[i]*vec[j]
        xy[i][j][1]=vec[i]*vec[j]
        xy[i][j][2]=vec[i]*vec[j]

for i in range(0,t2w.shape[0]):
    t2w[i,:,:,:]=t2w[i,:,:,:]*xy

for j in range(0,hbv.shape[0]):
    hbv[i,:,:,:]=hbv[i,:,:,:]*xy
    
for j in range(0,sag.shape[0]):
    sag[i,:,:,:]=sag[i,:,:,:]*xy

# merge couple list and corresponding target
data=np.zeros((couple.shape[0],4))
data_val=np.zeros((couple_val.shape[0],4))
data[:,0:2]=couple
data[:,2]=target
data_val[:,0:2]=couple_val
data_val[:,2]=target_val
data=data.astype(int)
data_val=data_val.astype(int)

# define euclidean distance
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

# define contrastive loss
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean((1 - y_true) * sqaure_pred + y_true * margin_square)

# define accuracy: not used for result
def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

# network for T2W modality
input_shapet2w = (t2w.shape[1], t2w.shape[2], t2w.shape[3])
left_inputt2w = Input(input_shapet2w)
right_inputt2w = Input(input_shapet2w)
convnet=Sequential()
convnet.add(Conv2D(8,(5,5),padding='same',activation='relu',input_shape=input_shapet2w))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(16,(3,3),padding='same',activation='relu'))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(32,(3,3),padding='same',activation='relu'))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(64,(3,3),padding='same',activation='relu'))
convnet.add(MaxPooling2D()) # added
convnet.add(Conv2D(96,(3,3),padding='same',activation='relu')) # added
convnet.add(BatchNormalization())
convnet.add(MaxPooling2D())
convnet.add(Flatten())
convnet.add(Dropout(0.3))
convnet.add(Dense(32,activation="sigmoid",name='flat')) # 32'''

# encode query and reference images (T2W)
encoded_lt2w = convnet(left_inputt2w)
encoded_rt2w = convnet(right_inputt2w)

# network for HBV modality
input_shapehbv = (hbv.shape[1], hbv.shape[2], t2w.shape[3])
left_inputhbv = Input(input_shapehbv)
right_inputhbv = Input(input_shapehbv)
convnet2=Sequential()
convnet2.add(Conv2D(8,(5,5),padding='same',activation='relu',input_shape=input_shapehbv))
convnet2.add(MaxPooling2D())
convnet2.add(Conv2D(16,(3,3),padding='same',activation='relu'))
convnet2.add(MaxPooling2D())
convnet2.add(Conv2D(32,(3,3),padding='same',activation='relu'))
convnet2.add(MaxPooling2D())
convnet2.add(Conv2D(64,(3,3),padding='same',activation='relu'))
convnet2.add(MaxPooling2D()) # added
convnet2.add(Conv2D(96,(3,3),padding='same',activation='relu')) # added
convnet2.add(BatchNormalization())
convnet2.add(MaxPooling2D())
convnet2.add(Flatten())
convnet2.add(Dropout(0.3))
convnet2.add(Dense(32,activation="sigmoid",name='flat')) # 32'''

# encode query and reference images (HBV)
encoded_lhbv = convnet2(left_inputhbv)
encoded_rhbv = convnet2(right_inputhbv)

# network for SAG images
input_shapesag = (sag.shape[1], sag.shape[2], sag.shape[3])
left_inputsag = Input(input_shapesag)
right_inputsag = Input(input_shapesag)
convnet3=Sequential()
convnet3.add(Conv2D(8,(5,5),padding='same',activation='relu',input_shape=input_shapesag))
convnet3.add(MaxPooling2D())
convnet3.add(Conv2D(16,(3,3),padding='same',activation='relu'))
convnet3.add(MaxPooling2D())
convnet3.add(Conv2D(32,(3,3),padding='same',activation='relu'))
convnet3.add(MaxPooling2D())
convnet3.add(Conv2D(64,(3,3),padding='same',activation='relu'))
convnet3.add(MaxPooling2D()) # added
convnet3.add(Conv2D(96,(3,3),padding='same',activation='relu')) # added
convnet3.add(BatchNormalization())
convnet3.add(MaxPooling2D())
convnet3.add(Flatten())
convnet3.add(Dropout(0.3))
convnet3.add(Dense(32,activation="sigmoid",name='flat')) # 32'''

# encode query and reference images (SAG)
encoded_lsag = convnet3(left_inputsag)
encoded_rsag = convnet3(right_inputsag)

# concatenate corresponding embedding for query and reference
concl=concatenate([encoded_lhbv,encoded_lt2w,encoded_lsag], axis=-1)
concl=Dense(32,activation="sigmoid")(concl)
concr=concatenate([encoded_rhbv,encoded_rt2w,encoded_rsag], axis=-1)
concr=Dense(32,activation="sigmoid")(concr)

# compute distance
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([concl, concr])

# define model, optimizer, loss
siamese_net = Model(input=[left_inputt2w,right_inputt2w,left_inputhbv,right_inputhbv,left_inputsag,right_inputsag],output=distance)
adam = Adam(0.00001)
siamese_net.compile(loss=contrastive_loss, optimizer=adam, metrics=[accuracy])

print(siamese_net.summary())

# TRAIN
loss=0
accuracy=0
best_loss=100

# iterate on epochs
for j in range(0,epochs):
    print(j)
    np.random.shuffle(data)

    # iterate on batches
    for i in range(0,data.shape[0]//batch_size):

        # get batch and train
        batch1t2w=t2w[data[i*batch_size:(i+1)*batch_size,0]]
        batch2t2w=t2w[data[i*batch_size:(i+1)*batch_size,1]]
        batch1hbv=hbv[data[i*batch_size:(i+1)*batch_size,0]]
        batch2hbv=hbv[data[i*batch_size:(i+1)*batch_size,1]]
        batch1sag=sag[data[i*batch_size:(i+1)*batch_size,0]]
        batch2sag=sag[data[i*batch_size:(i+1)*batch_size,1]]
        target_batch=data[i*batch_size:(i+1)*batch_size,2]
        c=siamese_net.train_on_batch([batch1t2w, batch2t2w, batch1hbv, batch2hbv, batch1sag,batch2sag], target_batch)

        # accumulate loss and accuracy
        loss=loss+c[0]
        accuracy=accuracy+c[1]

    # print train loss and accuracy
    print("loss:"+str(loss/(data.shape[0]//batch_size)))
    print("accuracy"+str(accuracy/(data.shape[0]//batch_size)))

    # evaluate loss and accuracy on the validation set and eventually save the the model
    loss_val, acc_val=siamese_net.evaluate([t2w[data_val[:,0]],t2w[data_val[:,1]],
                                                    hbv[data_val[:,0]],hbv[data_val[:,1]],
                                                   sag[data_val[:,0]],sag[data_val[:,1]]], data_val[:,2], verbose=0)
    if loss_val<best_loss:
        best_loss=loss_val
        siamese_net.save('SiameseTripleMax.h5')
    print("loss_val:"+str(loss_val))
    print("accuracy_val:"+str(acc_val))

    # reset training loss and accuracy
    loss=0
    accuracy=0

# reload model
del siamese_net
output_model=load_model("SiameseTriplemax.h5",custom_objects={'contrastive_loss': contrastive_loss})

# load couple and target for test data
target=np.load("./Couple/FirstRun/Alltarget_test.npy")
couple=np.load("./Couple/FirstRun/Allcouple_test.npy")

# predict embedding for the test data and save it
np.save("output_TripleMax.npy",output_model.predict([t2w[couple[:,0]],t2w[couple[:,1]],
                                                      hbv[couple[:,0]],hbv[couple[:,1]],
                                                          sag[couple[:,0]],sag[couple[:,1]]]))
