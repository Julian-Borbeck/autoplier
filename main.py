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

from sklearn import preprocessing
from sklearn.decomposition import PCA

# Globals
seed = 1017

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
Xtrain = np.log(Xtrain)
Xvalid = np.log(Xvalid)
#define a standard scaler to sclae the omics data
scaler = preprocessing.StandardScaler()
#Fit the scaler on the training data and scale the training data
Xtrain = scaler.fit_transform(Xtrain)
#scale validation data with the fit  scaler
Xvalid = scaler.transform(Xvalid)

#tranform Xtrain into Pi representation
Xtilde = np.dot(Xtrain,Cmat.to_numpy())
Xvalidtilde = np.dot(Xvalid,Cmat.to_numpy())
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

# init number of PCs to use
nPC = 10

#print cumulative variance explained in first nPC modes
cum_var = np.cumsum(pca.explained_variance_)
print(cum_var[0:nPC]/totvar)

#set number of latent variables to nPC
nz = nPC
nx = Xtilde.shape[1]
ny = nx
dropout_rate = .5

from keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
#from keras.utils.vis_utils import plot_model
from matplotlib import pyplot
from keras.callbacks import EarlyStopping, TensorBoard
from keras.regularizers import l1

# define encoder
visible = Input(shape=(nx,))
# encoder
encoder = Dense(nz, kernel_regularizer=l1(1E-4))(visible)
encoder = BatchNormalization()(encoder)
encoder = LeakyReLU()(encoder)
encoder = Dropout(dropout_rate)(encoder)
# decoder
decoder = Dense(ny, kernel_regularizer=l1(1E-4))(encoder)
decoder = BatchNormalization()(decoder)
decoder = LeakyReLU()(decoder)
encoder = Dropout(dropout_rate)(encoder)

# define autoencoder model
model = Model(inputs=visible, outputs=decoder)

# compile autoencoder model
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# plot the autoencoder
#plot_model(model, 'autoencoder.png', show_shapes=True)
model.summary()


################################
#Modelcheckpoint
#checkpointer = tf.keras.callbacks.ModelCheckpoint('bestmodel.h5', verbose=1, save_best_only=True)

callbacks = [
        EarlyStopping(patience=50, monitor='val_loss'),
        TensorBoard(log_dir='logs')]

####################################

# fit the autoencoder model to reconstruct input
history = model.fit(Xtilde, Xtilde, epochs=2000, batch_size=50, verbose=2,
                    validation_split=0.3, callbacks=callbacks)

# plot loss
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# define an encoder model (without the decoder)
final_encoder = Model(inputs=visible, outputs=encoder)
# save the encoder to file
final_encoder.save('encoder.h5')



#Define autoencoder
#from sknn.ae import AutoEncoder, Layer
#ae = AutoEncoder(
#    layers=[
#        Layer(activation="Sigmoid", warning=None, type=u'autoencoder', name="umat",
#              units=10, cost=u'msre', tied_weights=True, corruption_level=0.5)],
#        warning=None, parameters=None, random_state=42, learning_rule=u'sgd',
#        learning_rate=0.01, learning_momentum=0.9, normalize='batch',
#        regularize="L1", weight_decay=1E-4, dropout_rate=.5, batch_size=30,
#        n_iter=100, n_stable=10, f_stable=0.001, valid_set=None, valid_size=.3,
#        loss_type='mse', callback=None, debug=False, verbose=None)

#train AutoEncoder
#ae.fit(X_train)
