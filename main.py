""" generate metabolite data in latent variable representation """

#----------------------
# Nomenclature:
#
# (big)Omega are the omic features (metabolites)
# omega is number of omic features
#
# (big)M are the observations (urine samples)
# m is number of observations
#
# (big)Pi are the prior knowlegde of known pathways and metabolite classes
# pi is the number of the known pathways and metabolite classes.
#   (determined by HMDB for each metabolite in the dataset)
#
# (big)Lambda are the latent variables which are collective variables of the
#   known pathways and metabolite classes (PI)
# lambda is the number of latent variables
#
#----------------------
# Data Matricies:
#
# X are the observations in the native omic (metabolite) representation
#   with shape(m, omega)
#
# C is a binary membership matrix of shape (omega, pi) that describes which
#   metabolites belongs to each known pathways and metabolite class.
#
# Z are the observations in the latent variable representation
#   with shape (m, lambda)
#
# U is the transformation matrix that changes from the known pathway and
#   metabolite class representation to the latent variable representation with
#   shape (pi, lambda)
#
#----------------------
# Objective is to transform the native omics observations into a latent variable
# representation that is interpretable in terms of known pathways and metabolite
# classes.
#
#----------------------
# Approach adapted from https://doi.org/10.1038/s41592-019-0456-1:
#
# Compute a latent variable set that minimizes the reconstruction error of the
# native omics observations using an autoencoder (X -> <encoder> -> Z -> <decoder> Xhat).
#
# By tranforming X into the PI reperesentation the encoder behaves like U i.e.
# (XC) -> <logistic(U)> -> Z -> <logistic(U^T)> -> (XC)hat.
# The encoder and decoder are tied with logistic activation such that Z represents
# positive contributions of known pathways and metabolite classes in each observation.
#
# The interpretability of Z depends on the sparsity of U (i.e Z is dificult to
#   interpret if each observation has contributions from many known pathways and
#   metabolite classes.
# To this end L1 regularization and drop out is employed to encourage parsimony
#   effectively driving sparsity of the U matrix.
#
# Performance metrics (to minimize) are then
#   The reconstruction error: MSE[(XC), (XC)hat)]  (the training loss funciton)
#   The U sparsity: p1 norm of U (i.e. ||U||_1)
#   The Z parsimony: Frobenius norm of Z (i.e. ||Z||^2_F)
#
# Employ a grid search to optimize the performance metrics to hyperparameterize.
#----------------------

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn import preprocessing
from sklearn.decomposition import PCA

from keras.layers import Input, Dense, Dropout, PReLU, LeakyReLU, BatchNormalization
from keras.initializers import Constant
from keras.regularizers import l1
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, LambdaCallback

from matplotlib import pyplot

# Globals
seed = 1017
np.random.seed(seed)
tf.random.set_seed(seed)

#get membership data
membdf=pd.read_csv('data/membership.csv', sep=',')
#print(membdf.columns)

#get omics data
omdf=pd.read_csv('data/Training and validation combined and batch corrected_V3_20190827.csv', sep=',')
#print(omdf.columns)

#get clinical data
clindf=pd.read_csv('data/selective clinical data.csv', sep=',')
#print(clindf.columns)

#partition observations by subject id
trainids = clindf.loc[clindf["training_or_validation"]=='training']["subject_id"]
validids = clindf.loc[clindf["training_or_validation"]=='validation']["subject_id"]
#print(trainids)
#print(validids)

#get Xtrain and Xvalid
Xtrain = omdf.loc[omdf["subject_id"].isin(trainids), omdf.columns[2:]]
Xvalid = omdf.loc[omdf["subject_id"].isin(validids), omdf.columns[2:]]
#print(Xtrain)
#print(Xvalid)

#remove inf
Xtrain = Xtrain.replace( np.inf , np.nan )
Xvalid = Xvalid.replace( np.inf , np.nan )

#replace NA with half the minimum
Xtrain = Xtrain.replace(np.nan, np.min(Xtrain, axis=0)/2)
Xvalid = Xvalid.replace(np.nan, np.min(Xvalid, axis=0)/2)
#print(Xtrain.isnull().sum())
#print(Xtrain.isnull().sum().sum())


#assert all omics are in the membership data
missingomics =  Xtrain.columns[~(Xtrain.columns.isin(membdf["Standardized.name"])
                                | Xtrain.columns.isin(membdf["Input.name"]))]
if len(missingomics)>2:
    # WARNING: missing omics
    print("the foillowing omics were not found in the membership data.")
    print(missingomics)

#get the standardized omics labels for the columns in X
bigomega = [ membdf.loc[membdf["Input.name"] == metabo, "Standardized.name"].iloc[0]
             if metabo in membdf["Input.name"].values else metabo for metabo in Xtrain.columns]
#print(bigomega)

#order membership data by the omics order (columns of X)
Cmat = membdf.set_index("Standardized.name")
Cmat = Cmat.reindex(bigomega).iloc[:, 8:]
#print(Cmat)

#get the known pathway and metbolite classes (Pi) order
bigpi = Cmat.columns
#print(bigpi)

#log transform the omics data
X_train = np.log(Xtrain)
X_valid = np.log(Xvalid)
#define a standard scaler to sclae the omics data
scaler = preprocessing.StandardScaler()
#Fit the scaler on the training data and scale the training data
X_train = scaler.fit_transform(X_train)
#scale validation data with the fit  scaler
X_valid = scaler.transform(X_valid)

#tranform Xtrain into Pi representation
Xtilde = np.dot(X_train, Cmat.to_numpy())
Xvalidtilde = np.dot(X_valid, Cmat.to_numpy())
#print(Xtilde.shape)

# Full dof PCA for all data
pca = PCA(random_state = seed) #do not define number of PCs

#define standard scaler
scaler2 = preprocessing.StandardScaler()

#fit pca on the training data
X_scaled = scaler2.fit_transform(Xtilde)
X_pca = pca.fit_transform(X_scaled)

# compute total variation explained
totvar = sum(pca.explained_variance_)
cum_var = np.cumsum(pca.explained_variance_)/totvar

#show variation explained (diagnostic only)
nPC = 25
print(cum_var[0:nPC])


#hyperparams
dropout_rate = .4
regval = 2E-2#1.5E-2
patience = 100
batch_size = 50
maxepoch = 2000
valfrac = .3
pctexplained = .95#.997
alpha_init = .05


#set number of latent variables to nPC theat explain 99.7% of variance
nz = len(np.where(cum_var<=pctexplained)[0])
nx = Xtilde.shape[1]
ny = nx


# define encoder
visible = Input(shape=(nx,))
# encoder
ulayer = Dense(nz, kernel_regularizer=l1(regval))
encoder = ulayer(visible)
#encoder = BatchNormalization()(encoder)
#encoder = LeakyReLU()(encoder)
encoder = PReLU(alpha_initializer=Constant(value=alpha_init),
#encoder = PReLU(alpha_initializer=Constant(value=0.25),
alpha_regularizer='l1')(encoder)
encoder = Dropout(dropout_rate)(encoder)

# decoder
decoder = Dense(ny, kernel_regularizer=l1(regval))(encoder)
#decoder = Dense(ny)(encoder)
#decoder = BatchNormalization()(decoder)
decoder = LeakyReLU()(decoder)
#decoder = PReLU(alpha_initializer=Constant(value=alpha_init*5),
#alpha_regularizer='l1',
#alpha_constraint=None)(decoder)
#decoder = PReLU()(decoder)
decoder = Dropout(dropout_rate)(decoder)

# define autoencoder model
model = Model(inputs=visible, outputs=decoder)
# forbenius metric for Latent variables
model.add_metric(tf.math.reduce_sum(tf.math.square(encoder)), name='magz')
#model.add_metric(tf.math.reduce_max(tf.math.reduce_sum(ulayer.get_weights()[0], axis=0)), name='metric_1')
#print(ulayer.get_weights()[0].shape)
#print(tf.math.reduce_sum(ulayer.get_weights()[0], axis=0))

#custom metrics
#dependencies = {
#    'magz': magz
#}

# compile autoencoder model
model.compile(optimizer='adam', loss='mse')

# plot the autoencoder
#plot_model(model, 'autoencoder.png', show_shapes=True)
model.summary()


################################
#Modelcheckpoint
#checkpointer = tf.keras.callbacks.ModelCheckpoint('bestmodel.h5', verbose=1, save_best_only=True)

#from time import time
#from keras.callbacks import TensorBoard

logfile = open('logs/metrics.csv', mode='w')
logfile.write('epoch,umetric\n')
callbacks = [
        EarlyStopping(patience=patience, monitor='val_loss'),
        LambdaCallback(on_epoch_end=lambda epoch,
        logs: logfile.write(str(epoch)+","+str(tf.math.reduce_max(tf.math.reduce_sum(
        np.abs(ulayer.get_weights()[0]), axis=0)).numpy())+"\n"),
        on_train_end=lambda logs: logfile.close())]

####################################

# fit the autoencoder model to reconstruct input
history = model.fit(Xtilde, Xtilde, epochs=maxepoch, batch_size=batch_size, verbose=2,
                    validation_split=valfrac, callbacks=callbacks)

# plot loss
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# plot magnitude of latent variables
pyplot.plot(np.sqrt(history.history['magz']), label='||Z||_F train')
pyplot.plot(np.sqrt(history.history['val_magz']), label='||Z||_F test')
pyplot.legend()
pyplot.show()

# plot saved Umetric
udf=pd.read_csv('logs/metrics.csv', sep=',', encoding='utf-8' )
pyplot.plot(udf["epoch"], udf["umetric"], label='||U||_L1')
pyplot.legend()
pyplot.show()

#show U weights as image
fig, ax = pyplot.subplots(1,1)
w = np.array(ulayer.get_weights()[0])
img = ax.imshow(w, aspect='auto')
ax.set_yticks(range(len(bigpi)))
ax.set_yticklabels(bigpi)
fig.colorbar(img)
fig.subplots_adjust(left=0.7)
pyplot.show()

#show U weights as image
fig, ax = pyplot.subplots(1,1)
#find pathways(rows) with weight >.05
wtrim= w[np.any(w > 0.05, axis=1)]
bigpitrim = bigpi[np.any(w > 0.05, axis=1)]
img = ax.imshow(wtrim, aspect='auto')
ax.set_yticks(range(len(bigpitrim)))
ax.set_yticklabels(bigpitrim)
fig.colorbar(img)
fig.subplots_adjust(left=0.7)
pyplot.show()


# define an encoder model (without the decoder)
final_encoder = Model(inputs=visible, outputs=encoder)
# compile encoder model
final_encoder.compile(optimizer='adam', loss='mse')

# ----- save trained models ------
#save the autoencoder to file
model.save('autoencoder.h5')
# save the encoder to file
final_encoder.save('encoder.h5')

# ----- load saved models ------
#recover saved model
final_encoder = load_model('encoder.h5')
#load saved autoencoder
final_model = load_model('autoencoder.h5')

# ---- change basis for training and validation sets---

Ztrain = pd.DataFrame(final_encoder.predict(Xtilde), index=Xtrain.index)
Zvalid = pd.DataFrame(final_encoder.predict(Xvalidtilde), index=Xvalid.index)

#save transformed omics
Ztrain.to_csv('data/Ztrain.csv')
Zvalid.to_csv('data/Zvalid.csv')
