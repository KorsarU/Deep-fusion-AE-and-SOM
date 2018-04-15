    #from __future__ import division, print_function, absolute_import
from D_learn import autoencoder, features_comb, SOM
#from sklearn import preprocessing
#import tensorflow as tf
#import numpy as np
#import sklearn
#import pandas as pd


print("Training Geometric Features autoencoder, extrating hidden representation")
GF_output, GF_hidden_representation = autoencoder(0.01, 550, 1, 2, 20, 20, 'features_vectors/train/GF.csv', None)
print("Training LBP Features autoencoder, extrating hidden representation")
LBPF_output, LBPF_hidden_representation = autoencoder(0.01, 550, 1, 2, 170, 236, 'features_vectors/train/LBPF.csv', None)
print("Concatinating LBP and Geometric Features, passing concatinated data to the fusion autoencoder")
CF = features_comb(LBPF_hidden_representation,GF_hidden_representation)
print("Training fusion autoencoder, extrating hidden representation")
fusedF_output, fusedF_hidden_representation = autoencoder(0.01, 700, 1, 2, 170, 190, None, CF)

print("Feeding hidden representation of fused features to SOM classifier")
print("Initializing SOM classifier")
som = SOM(1, 2, 170, 400)
print("Training SOM classifier")
som.train(fusedF_hidden_representation)
print("Finished")