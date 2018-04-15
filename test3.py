from D_learn import autoencoder, features_comb, SOM
import numpy as np
from os import walk

def fuse(path):
    print("Training Geometric Features autoencoder, extrating hidden representation")
    GF_output, GF_hidden_representation = autoencoder(0.01, 550, 1, 2, 20, 20
                                                      , path+"GF.csv", None)
    
    print("Training LBP Features autoencoder, extrating hidden representation")
    LBPF_output, LBPF_hidden_representation = autoencoder(0.01, 550, 1, 2, 170, 236
                                                          , path+"LBPF.csv", None)
    
    print("Concatinating LBP and Geometric Features, passing concatinated data to the fusion autoencoder")
    
    CF = features_comb(LBPF_hidden_representation,GF_hidden_representation)
    
    print("Training fusion autoencoder, extrating hidden representation")
    fusedF_output, fusedF_hidden_representation = autoencoder(0.01, 700, 1, 2, 170, 190, None, CF)

    np.savetxt(path+"FusedOut.csv", fusedF_output, delimiter=",")
    np.savetxt(path+"FusedHiddenRP.csv", fusedF_hidden_representation, delimiter=",")

def start_fusion(path):
    count=-1
    dirs=None
    print(dirs)
    for subdir, dirss, files in walk(path):
            if count==-1:
                dirs=dirss
                count+=1
            else:
                fuse(subdir+"/"+dirs[count])
                count+=1

start_fusion("pics/mmi")
#start_fusion("pics/AFEW")
#start_fusion("pics/ck")
#start_fusion("pics/jaffe")
#start_fusion("pics/Stirling")