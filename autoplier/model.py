import pandas as pd
import numpy as np

from tensorflow.random import set_seed
from tensorflow.math import reduce_max, reduce_sum, square
from tensorflow.keras.layers import Input, Dense, Dropout, ReLU
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from sklearn import preprocessing


def set_seed_(seed):
    np.random.seed(seed)  # numpy seed
    set_seed(seed)  # tensorflow seed


class autoPLIER:

    def __init__(self, n_components=100, regval=1.20E-3):

        self.n_inputs = 2

        self.n_components = n_components

        self.regval = regval

        self.scaler = preprocessing.StandardScaler()

        self.components_decomposition_ = None

    def build_model(self):

        # - - - - - - Model Arch  - - - - - -
        # visible is the input data
        self.visible = Input(shape=(self.n_inputs,))

        # define a dense single layer (Ulayer) with L1 regularization to encourage sparsity
        # ulayer = Dense(nz, kernel_regularizer=l1(regval), activation="relu", name="ulayer")
        self.ulayer = Dense(self.n_components, kernel_regularizer=l1(self.regval), name="ulayer")

        # foward pass the input through the ulayer
        self.encoder = self.ulayer(self.visible)

        # Apply a ReLU type activation to constrain for positive weights
        # Logistic activation may also be a viable choice here - should give standardized
        #   latent variable values so we can skip a post-processing step.
        self.encoder = ReLU()(self.encoder)

        # The decoder does not have to be symmetric with encoder but let's have L1 reg anyway
        self.decoder = Dense(self.n_inputs)(self.encoder)

        # Apply a ReLU type activation
        self.decoder = ReLU()(self.decoder)

        # - - - - - - Build Model - - - - - -
        self.model = Model(inputs=self.visible, outputs=self.decoder)

        # Define a forbenius metric for the Latent variables to compare with paper
        self.model.add_metric(reduce_sum(square(self.encoder)), name='magz')

        # compile autoencoder model - with adam opt and use mse as reconstruction error
        self.model.compile(optimizer='adam', loss='mse')

    # - - - - - - Model Training  - - - - - -
    def fit(self, x_train, pathways, callbacks =[], batch_size=50, maxepoch=2000, verbose=2, valfrac=.3):

        x_train_processed = self.preprocess(x_train, pathways, fit = True)

        # fit the autoencoder model to reconstruct input
        history = self.model.fit(x_train_processed, x_train_processed, epochs=maxepoch, batch_size=batch_size, verbose=verbose,
                                 validation_split=valfrac, callbacks=callbacks)


        self.build_encoder()
        comp_dec = self.final_encoder.get_layer('ulayer')
        self.components_decomposition_ = pd.DataFrame(comp_dec.get_weights()[0], index= pathways.index)

        return history

    # - - - - - - Build Encoder Model - - - - - -
    def build_encoder(self):

        # define an encoder model (without the decoder)
        self.final_encoder = Model(inputs=self.visible, outputs=self.encoder)

        # compile encoder model- with adam opt and use mse as reconstruction error
        self.final_encoder.compile(optimizer='adam', loss='mse')

    def transform(self, x_predict, pathways):

        x_predict_processed = self.preprocess(x_predict, pathways, fit = False)

        z_predicted = pd.DataFrame(self.final_encoder.predict(x_predict_processed), index=x_predict.index)

        return z_predicted

    def fit_transform(self, x_train, pathways, callbacks=[], batch_size=50, maxepoch=2000, verbose=2, valfrac=.3):
        # fit the autoencoder model to reconstruct input

        x_train_processed = self.preprocess(x_train, pathways, fit = True)

        self.model.fit(x_train_processed, x_train_processed, epochs=maxepoch, batch_size=batch_size, verbose=verbose,
                                 validation_split=valfrac, callbacks=callbacks)

        z_predicted = pd.DataFrame(self.final_encoder.predict(x_train_processed), index=x_train.index)

        return z_predicted

    def preprocess(self, X, pathways, fit):

        X = X[X.columns[X.columns.isin(pathways.columns)]]
        pathways = pathways[pathways.columns[pathways.columns.isin(X.columns)]]

        X_tilde = np.dot(X, pathways.T.to_numpy())

        if(fit == True):
            X_tilde = self.scaler.fit_transform(X_tilde)
            self.n_inputs = X_tilde.shape[1]
            self.build_model()
            self.scaler_is_fit = True

        else:
            X_tilde = self.scaler.transform(X_tilde)

        return X_tilde
