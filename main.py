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
#   with shape(omega, m)
#
# C is a binary membership matrix of shape (omega, pi) that describes which
#   metabolites belongs to each known pathways and metabolite class.
#
# Z are the observations in the latent variable representation
#   with shape (lambda, m)
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
# (C^T X) -> <logistic(U^T)> -> Z -> <logistic(U)> -> (C^T X)hat.
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
#   The reconstruction error: MSE[(C^T X), (C^T X)hat)]  (the training loss funciton)
#   The U sparsity: p1 norm of U (i.e. ||U||_1)
#   The Z parsimony: Frobenius norm of Z (i.e. ||Z||^2_F)
#
# Employ a grid search to optimize the performance metrics to hyperparameterize.
#----------------------

#
#get metabolite observations X in R(omega, m)
