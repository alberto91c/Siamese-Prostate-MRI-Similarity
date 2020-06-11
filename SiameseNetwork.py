
# coding: utf-8

import numpy as np
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D, BatchNormalization, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.models import load_model
from keras.applications import MobileNet

# set batch size and epochs
batch_size=64
epochs=60

# load couple index and target of each couple
target=np.load("./Couple/ThirdRun/Alltarget.npy")
couple=np.load("./Couple/ThirdRun/Allcouple.npy")
target_val=np.load("./Couple/ThirdRun/AlltargetVal.npy")
couple_val=np.load("./Couple/ThirdRun/Allcouple_Val.npy")

# load data and scale
roi=np.load("./Data/roihbvSmall.npy")
roi=roi/np.max(roi)

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

for i in range(0,roi.shape[0]):
    roi[i,:,:,:]=roi[i,:,:,:]*xy

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

# network
input_shape = (roi.shape[1], roi.shape[2], roi.shape[3])
left_input = Input(input_shape)
right_input = Input(input_shape)
convnet=Sequential()
convnet.add(Conv2D(8,(5,5),padding='same',activation='relu',input_shape=input_shape))
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

# encode query and reference images
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)

# compute distance
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_l, encoded_r])

# define model, optimizer, loss
siamese_net = Model(input=[left_input,right_input],output=distance)
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
        batch1=roi[data[i*batch_size:(i+1)*batch_size,0]]
        batch2=roi[data[i*batch_size:(i+1)*batch_size,1]]
        target_batch=data[i*batch_size:(i+1)*batch_size,2]
        c=siamese_net.train_on_batch([batch1, batch2], target_batch)

        # accumulate loss and accuracy
        loss=loss+c[0]
        accuracy=accuracy+c[1]

    # print train loss and accuracy
    print("loss:"+str(loss/(data.shape[0]//batch_size)))
    print("accuracy"+str(accuracy/(data.shape[0]//batch_size)))

    # evaluate loss and accuracy on the validation set and eventually save the the model
    loss_val, acc_val=siamese_net.evaluate([roi[data_val[:,0]],roi[data_val[:,1]]], data_val[:,2],verbose=0)
    if loss_val<best_loss:
        best_loss=loss_val
        siamese_net.save('SiameseHBV.h5')
    print("loss_val:"+str(loss_val))
    print("accuracy_val:"+str(acc_val))

    # reset training loss and accuracy
    loss=0
    accuracy=0

# reload model
del siamese_net
output_model=load_model("SiameseHBV.h5",custom_objects={'contrastive_loss': contrastive_loss})

# load couple and target for test data
target=np.load("./Couple/ThirdRun/Alltarget_test.npy")
couple=np.load("./Couple/ThirdRun/Allcouple_test.npy")

# predict embedding for the test data and save it
np.save("output_HBV.npy",output_model.predict([roi[couple[:,0]], roi[couple[:,1]]]))


