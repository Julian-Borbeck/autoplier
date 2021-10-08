# CellO-MultiPLIER
Embed Omics data using an interpretable Tensorflow based model.

![Build Status](https://github.com/dmontemayor/jdrfmetabo/tree/devel/tests/badge.svg)

## Overview
This package contains a Tensorflow model built in spirit of the *PLIER* <sup>1</sup> approach to embed Omics data into a latent space representation.

## Installation
pip install -U git+https://github.com/dmontemayor/jdrfmetabo.git@devel

## Issues and Bug Reports

Please open an issue if you find any errors or unexpected behavior. Please make sure to document:

1. What you tried
2. What the result was
3. What you expected the result to be
4. Steps (if any) which you took to resolve the issue and their outcomes.


## Requirements
The following packages are required to use Autoplier:
- numpy<sup>2</sup>
- pandas<sup>3</sup>
- sklearn<sup>4</sup>
- tensorflow<sup>5</sup>

## Usage

### Setting seed
It is recommended to set a seed when using this package for reproducability. This can be done by calling autoplier.model.set_seed_(seed)

### Create model
A model can be created by calling the constructor autoPLIER() from autoplier.model.

Parameters:
- n_inputs: dimensionality of your input data(number of samples)
- n_components: dimensionality of the latent space representation (number of latent variables, default 100)
- dropout_rate: dropout rate of the model, default 0.09
- regval: regularization value, default 1.20E-3
- alpha_init: default 0.05

### Fit
After the model was created it can be fit using autoplier.model.fit() 

Parameters:
- x_train: training Dataset used to train the Model.
- callbacks: callbacks for the machine learning Model. default none
- batch_size: default 50
- maxepoch: default 2000
- verbose: default 2
- valfrac: default 0.3

### Transform
Transform new data into the latent space representation. Before transformation an encoder needs to be created using autoplier.model.build_encoder().
After an encoder was built new data can be transformed using autoplier.model.transform().

Parameters:
- x_predict: Dataset to be transformed into the latent space representation.
- index: name of the samples used in x_predict.

## Citations
1) Jaclyn N. Taroni, Peter C. Grayson, Qiwen Hu, Sean Eddy, Matthias Kretzler, Peter A. Merkel, Casey S. Greene, MultiPLIER: A Transfer Learning Framework for Transcriptomics Reveals Systemic Features of Rare Disease, Cell Systems, Volume 8, Issue 5, 2019, Pages 380-394.e4, https://doi.org/10.1016/j.cels.2019.04.003.

2) Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 2020, 357–362, DOI: 0.1038/s41586-020-2649-2.

3) Jeff Reback, Wes McKinney, jbrockmendel, Joris Van den Bossche, Tom Augspurger, Phillip Cloud, … Mortada Mehyar, pandas-dev/pandas: Pandas 1.0.3 (Version v1.0.3), 2020, http://doi.org/10.5281/zenodo.3715232

4)  Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 12, 2011, Pages 2825-2830

5)  Martín Abadi et. al, TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.