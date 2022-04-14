import os
import sys
import numpy as np
import pandas as pd
import random
import warnings
import glob
import netCDF4 as nc
import tensorflow as tf
from tensorflow import keras

# from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Permute, Reshape 
from tensorflow.keras.layers import Conv1D, SeparableConv2D, LeakyReLU, Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.utils import normalize
from tensorflow.keras.regularizers import l2
import tensorflow.keras.layers as kl
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import mean_squared_error, Precision, Recall
from tensorflow.keras.callbacks import CSVLogger

from numpy.random import seed, randint, rand, uniform

## Helper functions and constant variables

INPATH = "../input/pandora/data/full_raw/"
OUTPATH = "../working/"
SPLIT = [0.7, 0.8]  # Splitting ratio 0.70-0.10-0.20
BCHSIZE = 1000  # Batch size
EPOCH = 100  # Epochs
LR = 1e-5 # Learning rate
NCLASS = 2
TARGET = 'aero'

if (TARGET == "aerosol") | (TARGET == "aero"):
    filters = {'dof': [0.3626, 0.5157],        # scaled dof: 25th percetile and medium
               'chisq': [0.0305, 0.0673]}      # scaled chisq: medium and 75 percentile
elif TARGET == "no2":
    filters = {'dof': [0.6717, 0.8027],        # scaled dof: 25th percetile and medium
               'chisq': [0.0165, 0.0303]}      # scaled chisq: medium and 75 percentile

def get_flag(dof, chisq, filters, num_classes=2):
    flag = -1
    if num_classes == 2:
        if ((dof >= filters['dof'][1]) & (chisq <= filters['chisq'][1])) | \
        ((dof >= filters['dof'][0]) & (chisq <= filters['chisq'][0])):
            flag = 1
        else:
            flag = 0
            
    elif num_classes == 3:
        if ((dof >= filters['dof'][1]) & (chisq <= filters['chisq'][1])) | \
        ((dof >= filters['dof'][0]) & (chisq <= filters['chisq'][0])):
            flag = 1
        elif (dof < filters['dof'][0]) | (chisq > filters['chisq'][1]):
            flag = 0
        else:
            flag = 2 

    return flag

def r2_keras(y_true, y_pred):
    y_t = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, 0), tf.float32))
    y_p = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, 0), tf.float32))
    SS_res = K.sum(K.square(y_t - y_p))
    SS_tot = K.sum(K.square(y_t - K.mean(y_t)))
    return 1 - SS_res / (SS_tot + K.epsilon())


class data_generator(Sequence):
    def __init__(self, xnames, ynames, batch_size):
        self.xnames = xnames
        self.ynames = ynames
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.xnames) / float(self.batch_size)).astype(int)

    def __getitem__(self, idx):
        batch_x = self.xnames[idx * self.batch_size: idx * self.batch_size + self.batch_size]
        batch_y = self.ynames[idx * self.batch_size: idx * self.batch_size + self.batch_size]

        d2m = np.stack([np.load(s)['d2m_scl'] for s in batch_x])   
        t2m = np.stack([np.load(s)['t2m_scl'] for s in batch_x])   
        skt = np.stack([np.load(s)['skt_scl'] for s in batch_x])   
        sp = np.stack([np.load(s)['sp_scl'] for s in batch_x])   
        sza = np.stack([np.load(s)['sza_scl'] for s in batch_x])   
        raa = np.stack([np.load(s)['raa_scl'] for s in batch_x])   
        o4 = np.stack([np.load(s)['o4_scl'] for s in batch_x])  

        tempx1 = np.stack([d2m, t2m, skt, sp, sza, raa, o4], axis=-1)
        
        tprof = np.stack([np.load(s)['tprof_scl'] for s in batch_x])   
        qprof = np.stack([np.load(s)['qprof_scl'] for s in batch_x])   
        uprof = np.stack([np.load(s)['uprof_scl'] for s in batch_x])   
        vprof = np.stack([np.load(s)['vprof_scl'] for s in batch_x])

        if (TARGET == "aerosol") | (TARGET == "aero"):
            tempx2 = np.stack([tprof, qprof, uprof, vprof], axis=-1)
            flag = [get_flag(np.load(s)['dof_aero_scl'], np.load(s)['chisq_aero_scl'], filters) 
                    for s in batch_y]
        
        elif TARGET == "no2":
            aeroprof = np.stack([np.load(s)['aero_scl'] for s in batch_x])
            tempx2 = np.stack([tprof, qprof, uprof, vprof, aeroprof], axis=-1)
            flag = [get_flag(np.load(s)['dof_no2_scl'], np.load(s)['chisq_no2_scl'], filters) 
                    for s in batch_y]
        
        flag = to_categorical(flag, num_classes=NCLASS)
        return [tempx1, tempx2], flag


def data_split(x, y, ratio1, ratio2, maskname=None):
    dsize1 = int(x.shape[0] * ratio1)
    dsize2 = int(x.shape[0] * ratio2)
    dmask = np.array(list(range(0, x.shape[0])))
    random.shuffle(dmask)

    dmask1 = dmask[:dsize1]
    x1 = x[dmask1]
    y1 = y[dmask1]

    dmask2 = dmask[dsize1:dsize2]
    x2 = x[dmask2]
    y2 = y[dmask2]

    dmask3 = dmask[dsize2:]
    x3 = x[dmask3]
    y3 = y[dmask3]

    if maskname:
        np.savez(maskname, train=dmask1, valid=dmask2, test=dmask3)

    return x1, y1, x2, y2, x3, y3


def get_x(fname):
    d2m = np.load(fname)['d2m_scl']    
    t2m = np.load(fname)['t2m_scl']    
    skt = np.load(fname)['skt_scl']    
    sp = np.load(fname)['sp_scl']    
    sza = np.load(fname)['sza_scl']    
    raa = np.load(fname)['raa_scl']    
    o4 = np.load(fname)['o4_scl']

    tempx1 = np.stack([d2m, t2m, skt, sp, sza, raa, o4], axis=-1)
    tempx1 = tempx1[np.newaxis, :, :]

    tprof = np.load(fname)['tprof_scl']    
    qprof = np.load(fname)['qprof_scl']    
    uprof = np.load(fname)['uprof_scl']    
    vprof = np.load(fname)['vprof_scl']
    
    if (TARGET == "aerosol") | (TARGET == "aero"):
        tempx2 = np.stack([tprof, qprof, uprof, vprof], axis=-1)
    elif TARGET == "no2":
        aeroprof = np.load(fname)['aero_scl']
        tempx2 = np.stack([tprof, qprof, uprof, vprof, aeroprof], axis=-1)

    tempx2 = np.stack([tprof, qprof, uprof, vprof], axis=-1)
    tempx2 = tempx2[np.newaxis, :, :]

    return [tempx1, tempx2]

def get_y(fname):
    if (TARGET == "aerosol") | (TARGET == "aero"):
        dof = np.load(fname)['dof_aero_scl']
        chisq = np.load(fname)['chisq_aero_scl']
    elif TARGET == "no2":
        dof = np.load(fname)['dof_no2_scl']
        chisq = np.load(fname)['chisq_no2_scl']
    return dof, chisq


## Input data preparation

preset_mask = True  # wether to load previous mask

xfiles = np.array(sorted(glob.glob(INPATH + 'X/X_*.npz')))
yfiles = np.array(sorted(glob.glob(INPATH + 'Y/Y_*.npz')))

if preset_mask:
    
    maskfile = "../input/pandora/masks/mask_cls.npz"
    
    mask1 = np.load(maskfile)['train']
    mask2 = np.load(maskfile)['valid']
    mask3 = np.load(maskfile)['test']   
    
    xlist1 = xfiles[mask1]
    ylist1 = yfiles[mask1]
    xlist2 = xfiles[mask2]
    ylist2 = yfiles[mask2]
    xlist3 = xfiles[mask3]
    ylist3 = yfiles[mask3]

else:
    xlist1, ylist1, \
    xlist2, ylist2, \
    xlist3, ylist3 = data_split(xfiles, yfiles, SPLIT[0], SPLIT[1],
                                maskname='mask_cls.npz') 

print('Splitting: ', 
      len(xlist1), len(ylist1), 
      len(xlist2), len(ylist2), 
      len(xlist3), len(ylist3))

train_generator = data_generator(xlist1, ylist1, BCHSIZE)   
valid_generator = data_generator(xlist2, ylist2, BCHSIZE) 
test_generator = data_generator(xlist3, ylist3, BCHSIZE)


## Model setup

if (TARGET == "aerosol") | (TARGET == "aero"):
    input_size = (21, 4)
elif TARGET == "no2":
    input_size = (21, 5)
    
if NCLASS == 2:
    loss = 'categorical_crossentropy'
    actvn = 'softmax'
    units = NCLASS
else:
    loss = 'categorical_crossentropy'
    actvn = 'softmax'
    units = NCLASS

prof_input = Input((input_size), name='rean_input')
meas_input = Input((9, 7), name='meas_input')

pc1 = Conv1D(36, 2, strides=1, padding='same', activation="relu")(meas_input)  # 9, 36
pc1 = Conv1D(36, 2, strides=1, padding='same', activation="relu")(pc1)         # 9, 36
pd1 = Dense(36, activation="relu")(pc1) # 9, 36
pc2 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(pd1)        # 9, 128
pc2 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(pc2)        # 9, 128
pd2 = Dense(128, activation="relu")(pc2) # 9, 128
pc3 = Conv1D(256, 2, strides=1, padding='same', activation="relu")(pd2)        # 9, 256
pc3 = Conv1D(256, 2, strides=1, padding='same', activation="relu")(pc3)        # 9, 256
pd3 = Dense(256, activation="relu")(pc3) # 9, 256

lstm = LSTM(21, name='LSTM1') (pd3)  # this gives a pseudo profile
lstm = kl.Reshape((21, 1)) (lstm)

concat_layer = concatenate([prof_input, lstm])   # 21, 5

c1 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(concat_layer)  # 21, 32

c1 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(c1)  # 21, 32
p1 = Dense(128, activation="relu") (c1)   # 21, 32

c2 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(p1)  # 21, 64
c2 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(c2)  # 21, 64
p2 = Dense(128, activation="relu") (c2)   # 21, 64

c3 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(p2) # 21, 128
c3 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(c3) # 21, 128
p3 = Dense(128, activation="relu") (c3)   # 21, 128


# define the classifier branch
meas_f = Flatten()(meas_input)
prof_f = Flatten()(prof_input)
pd1f = Flatten()(pd1)
pd2f = Flatten()(pd2)
pd3f = Flatten()(pd3)
p1f = Flatten()(p1)
p2f = Flatten()(p2)
p3f = Flatten()(p3)

classinput = concatenate([meas_f, prof_f, pd1f, pd2f, pd3f, p1f, p2f, p3f])  # 12012

ch1 = Dense(512, activation="sigmoid") (classinput)   # 512
ch2 = Dense(64, activation="sigmoid") (ch1)   # 64
classoutput = Dense(units, activation=actvn, name='QF_classifier') (ch2)   # 4

# define a model with a list of two inputs
model = Model(inputs=[meas_input, prof_input], outputs=[classoutput])

opt = Adam(learning_rate = LR)

model.compile(optimizer=opt, loss=loss, metrics=['accuracy', Precision(), Recall()])
model.summary()

csv_logger = CSVLogger('DLMAXDOASCLS_log.csv', append=True, separator=';')
earlystopper = EarlyStopping(patience=20, verbose=1)
checkpointer = ModelCheckpoint('DLMAXDOASCLS_checkpt_{val_loss:.2f}.h5',
                               verbose=1, save_best_only=True)

## Load weights from pre-trained model
ep_complete = 300
load_weights = True

if load_weights:
    model_path = "../input/pandora/saved_models/DLMAXDOASCLS_ep%s.h5" % (ep_complete)
    model.load_weights(model_path)

#     print("Evaluation of pre-trained model for train set:")
#     print(model.evaluate(train_generator, batch_size=BCHSIZE))

#     print("Evaluation of pre-trained model for val set:")
#     print(model.evaluate(valid_generator, batch_size=BCHSIZE))

    print("Evaluation of pre-trained model for test set:")
    print(model.evaluate(test_generator, batch_size=BCHSIZE))

## Model fitting

results = model.fit(train_generator,
                    validation_data=valid_generator, epochs=EPOCH,
                    callbacks=[earlystopper, checkpointer, csv_logger])
model.save('DLMAXDOASCLS_ep%s.h5' % (ep_complete+EPOCH))

# print("Evaluation of trained model for train set:")
# print(model.evaluate(train_generator, batch_size=BCHSIZE))

# print("Evaluation of trained model for val set:")
# print(model.evaluate(valid_generator, batch_size=BCHSIZE))

print("Evaluation of trained model for test set:")
print(model.evaluate(test_generator, batch_size=BCHSIZE))

## Save outputs

SaveOutput = True

if SaveOutput:

    sep = pd.DataFrame(columns=['Datetime', 'yprob', 'ypred', 'ytrue'])

    for i, xfile, yfile in zip(range(len(xlist3)), xlist3, ylist3):
        fname = os.path.split(xfile)[-1]
        time = fname.split('.')[0][2:-1]
        yprob = model.predict(get_x(xfile))[0]
        ypred = int(yprob[-1] > 0.5)
        dof, chisq = get_y(yfile)
        ytrue = get_flag(dof, chisq, filters)
        sep.loc[i] = [time, yprob, ypred, ytrue]
    
    sep.to_csv("testcls.csv", index=False)

