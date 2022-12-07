import os
import pandas as pd
import pickle
import numpy as np

from tensorflow.random import set_seed
from tensorflow.math import reduce_max, reduce_sum, square
from tensorflow.keras.layers import Input, Dense, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, average_precision_score
from sklearn.model_selection import GridSearchCV

def set_seed_(seed):
    np.random.seed(seed)  # numpy seed
    set_seed(seed)  # tensorflow seed

class autoPLIER:

    def __init__(self, n_components="estimate", regval=1.20E-7, learning_rate = 0.0002, scaler = None):

        self.n_inputs = 2

        self.n_components = n_components

        self.regval = regval

        self.scaler = scaler

        self.components_decomposition_ = None

        self.learning_rate = learning_rate

        self.scaled_components_decomposition_ = None

    def build_model(self):

        # - - - - - - Model Arch  - - - - - -
        # visible is the input data
        self.visible = Input(shape=(self.n_inputs,))

        # define a dense single layer (Ulayer) with L1 regularization to encourage sparsity
        # ulayer = Dense(nz, kernel_regularizer=l1(regval), activation="relu", name="ulayer")
        self.ulayer = Dense(self.n_components, kernel_regularizer=l1(self.regval), kernel_constraint=NonNeg(),
                            use_bias=False, name="ulayer")

        # foward pass the input through the ulayer
        self.encoder = self.ulayer(self.visible)

        # Apply a ReLU type activation to constrain for positive weights
        # Logistic activation may also be a viable choice here - should give standardized
        #   latent variable values so we can skip a post-processing step.
        self.encoder = ReLU()(self.encoder)

        # The decoder does not have to be symmetric with encoder but let's have L1 reg anyway
        self.decoder = Dense(self.n_inputs, kernel_constraint=NonNeg(), use_bias=False)(self.encoder)

        # Apply a ReLU type activation
        self.decoder = ReLU()(self.decoder)

        # - - - - - - Build Model - - - - - -
        self.model = Model(inputs=self.visible, outputs=self.decoder)

        # Define a forbenius metric for the Latent variables to compare with paper
        self.model.add_metric(reduce_sum(square(self.encoder)), name='magz')

        #Define optimizer and learning rate
        self.optimizer = Adam(learning_rate=self.learning_rate)

        # compile autoencoder model - with adam opt and use mse as reconstruction error
        self.model.compile(optimizer= self.optimizer, loss='mse')

    # - - - - - - Model Training  - - - - - -
    def fit(self, x_train, pathways, callbacks=[], batch_size=None, maxepoch=2000, verbose=2, valfrac=.3):

        x_train_processed = self.preprocess(x_train, pathways, fit=True)

        # fit the autoencoder model to reconstruct input
        history = self.model.fit(x_train_processed, x_train_processed, epochs=maxepoch, batch_size=batch_size,
                                 verbose=verbose, validation_split=valfrac, callbacks=callbacks)

        self.build_encoder()
        comp_dec = self.final_encoder.get_layer('ulayer')
        self.components_decomposition_ = pd.DataFrame(comp_dec.get_weights()[0], index=pathways.index)
        self.scaled_components_decomposition_ = (self.components_decomposition_-self.components_decomposition_.min())/\
                                                (self.components_decomposition_.max()-
                                                 self.components_decomposition_.min())
        return history

    # - - - - - - Build Encoder Model - - - - - -
    def build_encoder(self):
        # define an encoder model (without the decoder)
        self.final_encoder = Model(inputs=self.visible, outputs=self.encoder)

        # compile encoder model- with adam opt and use mse as reconstruction error
        self.final_encoder.compile(optimizer= self.optimizer, loss='mse')

    def transform(self, x_predict, pathways):

        x_predict_processed = self.preprocess(x_predict, pathways, fit=False)

        z_predicted = pd.DataFrame(self.final_encoder.predict(x_predict_processed), index=x_predict.index)

        return z_predicted

    def fit_transform(self, x_train, pathways, patience = None, batch_size=None, maxepoch=2000, verbose=2, valfrac=.3):

        if(patience != None):
            callbacks = [EarlyStopping(patience = patience)]
        else:
            callbacks = []
        # fit the autoencoder model to reconstruct input


        self.fit(x_train, pathways, maxepoch=maxepoch, batch_size=batch_size, verbose=verbose,
                       valfrac=valfrac, callbacks=callbacks)

        z_predicted = self.transform(x_train, pathways)

        return z_predicted

    def preprocess(self, X, pathways, fit):


        X = X[X.columns[X.columns.isin(pathways.columns)]]
        pathways = pathways[pathways.columns[pathways.columns.isin(X.columns)]]

        pathways = pathways[X.columns]

        X_tilde = np.dot(X, pathways.T.to_numpy())

        if fit is True:
            if (self.scaler):
                X_tilde = self.scaler.fit_transform(X_tilde)
            self.n_inputs = X_tilde.shape[1]
            self.build_model()
            self.scaler_is_fit = True

            if(self.n_components == "estimate"):
                self.n_components = get_n_LVs(X_tilde, seed = 111, Pct_exp_var = 0.95, m = 5 )

        else:
            if (self.scaler):
                X_tilde = self.scaler.transform(X_tilde)




        return X_tilde


    def get_top_pathways(self, LVs, n_pathways):
        pathwaydict = {}
        for LV in LVs:
            pathwaydict[LV] = self.components_decomposition_[LV].sort_values(ascending=False)[0:n_pathways]
        return pathwaydict


    def get_top_pathway_LVs(self, pathway, n_LVs):
        LVs = self.components_decomposition_.T[pathway].sort_values(ascending=False)[0:n_LVs]
        return LVs

# epsilon sparsity function
def sparsity_epsilon(z, epsilon):
    s = (np.sum((np.abs(z) < epsilon).astype(int)).sum()) / float(z.size)
    return s

# calculate number of LVs to use from inherent dimensionality of the data, calculated from the number of PCA
# components that explain 95 percent variance
def get_n_LVs(X_tilde, seed, Pct_exp_var = 0.95, m = 5 ):

    pca = PCA(random_state=seed)  # do not define number of PCs

    X_pca = pca.fit_transform(X_tilde)

    totvar = sum(pca.explained_variance_)
    cum_var = np.cumsum(pca.explained_variance_) / totvar

    for i, val in enumerate(cum_var):

        if (val >= Pct_exp_var):

            return i * m
    return 0

def optimize_l1(target_sparsity, delta, start_l1, x_train, pathways, callbacks=[],
                batch_size=None, maxepoch=2000, verbose=0, valfrac=.3, n_components=100, learning_rate = 0.0002):
    set_seed_(111)
    sparsity = 0
    tuning_l1 = start_l1
    step = 10
    closest = 1
    closest_l1 = tuning_l1
    while abs(sparsity - target_sparsity) > delta:

        mod = autoPLIER(regval=tuning_l1, n_components=n_components, learning_rate = learning_rate)
        mod.fit(x_train, pathways, callbacks, batch_size=batch_size,
                maxepoch=maxepoch, verbose=verbose, valfrac=valfrac)
        sparsity = sparsity_epsilon(mod.components_decomposition_, 10 ** -4)
        diff = sparsity - target_sparsity

        if abs(diff) < closest and diff < 0:
            closest = abs(diff)
            closest_l1 = tuning_l1
            if abs(diff) > delta:
                tuning_l1 = tuning_l1 * step

        else:

            if diff > 0:
                step = step / 2
                tuning_l1 = closest_l1 * step
            else:
                tuning_l1 = closest_l1 / step
                step = step / 2
                tuning_l1 = tuning_l1 * step
    return closest_l1


def get_top_LVs(sample_df, n_LVs):
    LV_dict = {}
    for index, row in sample_df.iterrows():
        sorted_samples = row.sort_values(ascending=False)[0:n_LVs]

        LV_dict[index] = sorted_samples
    return LV_dict


# fscore metric used to evaluate classifiers
def fscore(p, r):
    denom = p + r or 1

    return 2*(p * r) / denom


def train_classifiers(X_train, X_test, y_train, y_test):

    PARAMETERS = {
        'C': [
            0.001,
            0.01,
            0.1,
            1.0,
            10.0,
            100.0
            ]
        }
    MAX_ITER = 200000

    lr_model = LogisticRegression(penalty='l1', solver='liblinear')

    clf = GridSearchCV(lr_model, PARAMETERS, scoring='f1')

    clf.fit(X_train, y_train)

    best_params = max(
        zip(
            clf.cv_results_['param_C'],
            clf.cv_results_['mean_test_score']
            ),
        key=lambda x: x[1]
    )
    best_C = best_params[0]


    lr_model = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, max_iter=1000)
    lr_model.fit(X_train, y_train)

    target_pred = lr_model.predict(X_test)
    target_probs = lr_model.predict_proba(X_test)[:, 1]
    precision = precision_score(y_test, target_pred)
    recall = recall_score(y_test, target_pred)
    ap = average_precision_score(y_test, target_probs)
    f = fscore(precision, recall)


    return (f, ap)

def save_model(path, model_name, AP_instance):
    with open(os.join(path,model_name), 'wb') as pickle_file:
        pickle.dump(AP_instance, pickle_file)

def load_model(model_path):
    with open(model_path, 'rb') as pickle_file:
        AP_instance = pickle.load(pickle_file)
    return AP_instance