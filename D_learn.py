from __future__ import division, print_function, absolute_import
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import sklearn
import pandas as pd
from tqdm import tqdm

def get_label_vect(num):
    vect= np.zeros((7))
    vect[num-1]=1
    return vect

def get_num_from_vect(vect):
    for i in range(len(vect)):
        if vect[i]==1:
            return i+1
        
def autoencoder(learning_rate, training_epochs, display_step,
               alpha, n_hidden_nodes, n_input_nodes, data_path = None,
                concat_features = None):
# In order to pass in the data autoencoder may take data path argument - which is a path
#   to .csv file containing feature vectors, or concatinated features available as a 
#    variable. Each autoencoder can only take one out of two arguments, this is to say that 
#    once one argument is passed the other one is set to None """

    if data_path != None:
        df=pd.read_csv(data_path, sep=',',header=None)
        FT = df.values
        FT_norm = preprocessing.normalize(FT, axis=1)
    else:
        FT = concat_features
        FT_norm = preprocessing.normalize(FT, axis=1)
    
    # Setting input node od autoencoder
    
    X = tf.placeholder("float", [None, n_input_nodes])
    
    # Setting up weight and biases dictionary
    
    weights = {
        'encoder_w': tf.Variable(tf.random_normal([n_input_nodes, n_hidden_nodes])),
        'decoder_w': tf.Variable(tf.random_normal([n_hidden_nodes, n_input_nodes])),
    }
    biases = {
        'encoder_b': tf.Variable(tf.random_normal([n_hidden_nodes])),
        'decoder_b': tf.Variable(tf.random_normal([n_input_nodes])),
    }
    
    # Setting up encoding phase
    def encoder(x):
        # Get hidden layer activation values
        hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_w']),
                                            biases['encoder_b']))
        return hidden_layer
    
    # Setting decoding phase
    def decoder(x):
        # Decoding input layer with sigmoid activation 
        output_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_w']),
                                            biases['decoder_b']))
        return output_layer
    
    # Assembling autoencoder
    encoding = encoder(X)
    decoding = decoder(encoding)
    
    # Estimating input data
    X_est = decoding
    # True data to compare with
    X_true = X
    
    # Cost and optimization
    summ_square_errors = tf.reduce_mean(tf.pow(X_true - X_est, 2))
    # Weight in encoder part of autoencoder
    weight_decay_1 = tf.nn.l2_loss(weights['encoder_w'])
    # Weight in decoder part of autoencoder
    weight_decay_2 = tf.nn.l2_loss(weights['decoder_w'])
    cost = summ_square_errors + alpha * (weight_decay_1 + weight_decay_2) 
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    # Initializing the variables
    init = tf.initialize_all_variables()
    
    # Start the network
    sess = tf.InteractiveSession()
    sess.run(init)
    
    # Training
    
    for epoch in range(training_epochs):
        # Running backpropagation optimization
        _, c = sess.run([optimizer, cost], feed_dict={X: FT_norm})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))
            
    print("Network optimized")
    
    # Getting output and hidden layer values
    AE_output = sess.run(decoding, feed_dict={X: FT_norm})
    AE_hidden_representation = sess.run(encoding, feed_dict={X: FT_norm})
    
    #hidden-reprsentation normalize
    AE_hidden_representation = preprocessing.normalize(AE_hidden_representation, axis=0)
    
    return AE_output, AE_hidden_representation

# Fetures concatination

def features_comb(LBPF_hidden_representation,GF_hidden_representation):
    # Combined hidden representation of GF and LBPF
    _ = []
    for x,y in zip(LBPF_hidden_representation,GF_hidden_representation):
        cond = np.concatenate([x,y])
        _.append(cond)
        CF = np.array(_)
    
    return CF

# SOM classifier

class SOM(object):
  
 
    #To check if the SOM has been trained
    _trained = False
 
    def __init__(self, m, n, dim, n_iterations=100, alpha0=None,alphaf=None, sigma0=None, sigmaf=None, learning_rate=0.1):
        """
        SOM intialization:
 
        - m, n: dimensions of SOM 
        - 'n_iterations': number of iterations during training
        - 'dim': dimensionality of the training data 
        - 'alpha0,alphaf': initial and nth iteration values of alpha
        - 'sigma0,sigma0': is the the initial and nth iteration neighbourhood value, denoting
                   the radius of influence of the BMU while training. By default, its
                   taken to be half of max(m, n).
        """
 
        #Assign required variables first
        self._m = m
        self._n = n
        
        if alpha0 is None:
            alpha0 = 0.95
        else:
            alpha0 = float(alpha0)
        if alphaf is None:
            alphaf = 0.005
        else:
            alphaf = float(alphaf)
            
        if sigma0 is None:
            sigma0 = 3.5
        else:
            sigma0 = float(sigma0)
        if sigmaf is None:
            sigmaf = 0.001
        else:
            sigmaf = float(sigmaf)
            
        self._n_iterations = abs(int(n_iterations))
 
        ##INITIALIZE GRAPH
        self._graph = tf.Graph()
 
        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():
 
            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE
 
            #Randomly initialized weightage vectors for all neurons,
            #stored together as a matrix Variable of size [m*n, dim]
            self._weightage_vects = tf.Variable(tf.random_normal(
                [m*n, dim]),name= "wj")
            
            self.Aj = tf.Variable(tf.random_normal(
                [7,dim]),name= "Aj")
            
            self.bj = tf.Variable(tf.random_normal(
                [7,m*n]),name= "bj")
 
            #Matrix of size [m*n, 2] for SOM grid locations
            #of neurons
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))),name="location_vects")
            
            self.trainable = tf.placeholder("bool")
            ##PLACEHOLDERS FOR TRAINING INPUTS
            #We need to assign them as attributes to self, since they
            #will be fed in during training
 
            #The training vector
            
            #self._vect_input = tf.placeholder("float", [dim],name= "inputvector")
            
            #self.inputvect =tf.nn.l2_normalize(self._vect_input,0)
            self.inputvect =tf.placeholder("float", [dim],name= "inputvector")
            
            #Iteration number
            self._iter_input = tf.placeholder("float",name="itervector")
 
            self.y = tf.placeholder("float", [7],name="label")
            
            
            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            #Only the final, 'root' training op needs to be assigned as
            #an attribute to self, since all the rest will be executed
            #automatically during training
 
            #To compute the Best Matching Unit given a vector
            #Basically calculates the Euclidean distance between every
            #neuron's weightage vector and the input, and returns the
            #index of the neuron which gives the least value
            with tf.name_scope("bmu_index_extraction"):
                bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                        tf.pow(tf.subtract(tf.stack([self.inputvect for i in range(m*n)])
                        ,self._weightage_vects), 2), 1)),0)     
            #This will extract the location of the BMU based on the BMU's
            #index
                slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                     np.array([[0, 1]]))
                print(slice_input)
                tf.cast(slice_input, tf.int32)
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]), dtype=tf.int64)),
                                 [2])
     
            #To compute the alpha and sigma values based on iteration
            #number
            with tf.name_scope("alpha_and_sigma_eval"):
                self.iterfract= tf.div(self._iter_input,self._n_iterations)
                
                alphaNdiv0 = tf.div(alphaf,alpha0)
                alphapow = tf.pow(alphaNdiv0,self.iterfract)
                
                sigmaNdiv0 = tf.div(sigmaf,sigma0)
                sigmapow = tf.pow(sigmaNdiv0,self.iterfract)
                
                alphaN = tf.multiply(alpha0, alphapow, name="alphaN")
                sigmaN = tf.multiply(sigma0, sigmapow, name="sigmaN")
 
            #Construct the op that will generate a vector with learning
            #rates for all neurons, based on iteration number and location
            #wrt BMU.
            with tf.name_scope("bmu_distance_squares"):
                bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                    self._location_vects, tf.stack(
                        [bmu_loc for i in range(m*n)])), 2), 1)
   
            self.sigcheck=tf.div(tf.cast(
                    bmu_distance_squares, "float32"), tf.pow(sigmaN, 2))
            
            with tf.name_scope("neighbourhood_func"):
                self.neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                    bmu_distance_squares, "float32"), tf.pow(sigmaN, 2))))
                learning_rate_op = tf.multiply(alphaN, self.neighbourhood_func)
 
            #Finally, the op that will use learning_rate_op to update
            #the weightage vectors of all neurons based on a particular
            #input
            with tf.name_scope("som_weight_upd"):
                learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                    learning_rate_op, np.array([i]), np.array([1])), [dim])
                                                   for i in range(m*n)])
            with tf.name_scope("weightage_delta"):
                weightage_delta = tf.multiply(
                    learning_rate_multiplier,
                    tf.subtract(tf.stack([self.inputvect for i in range(m*n)]),
                           self._weightage_vects))                                         
                new_weightages_op = tf.add(self._weightage_vects,
                                           weightage_delta)
                
                self._som_w_upd= tf.cond(self.trainable,lambda: tf.assign(self._weightage_vects,
                                              new_weightages_op, name="train"),lambda: tf.add(sigma0,sigma0))                                     
            
            with tf.name_scope("x_sub_wj"):
                x_sub_wj=tf.subtract(self._weightage_vects, tf.stack(
                        [self.inputvect for i in range(m*n)]))

            with tf.name_scope("b_plus_a_x_sub_wj"):
                Zj=tf.add(tf.matmul(self.Aj,tf.transpose(x_sub_wj)),self.bj)
                
            
            with tf.name_scope("Z"):
                Z1=tf.reduce_sum(tf.multiply(self.neighbourhood_func,Zj),1)
                self.Z2=tf.reduce_sum(self.neighbourhood_func,0)
                Z=tf.div(Z1,self.Z2)
            
            self.sig=tf.sigmoid(Z,name="sigmoid")
            
            labels = tf.to_int64(self.y)
            
            self.softmax=tf.nn.softmax(self.sig)
            
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels, logits=self.sig, name='xentropy')

            self.loss = tf.reduce_mean(self.cross_entropy, name='xentropy_mean')
            
            tf.summary.scalar('loss', self.loss)
            
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train = optimizer.minimize(self.loss)
            
            #TODO: Histograms with diffferent learning rates on Tensorboard
            
            ##INITIALIZE SESSION
            self._sess = tf.Session()
            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)
 
    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        #Nested iterations over both dimensions
        #to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])
 
    def evaluate(self,samples,iternum):
        acc=0
        for i in range(samples.shape[0]):
            out=self._sess.run(self.softmax,
                               feed_dict={self.inputvect: samples[i][:170],
                                          self._iter_input: iternum,
                                           self.trainable: False})
            if (np.argmax(out)+1)==int(samples[i][170]):
                acc+=1
        if(samples.shape[0]==0):
            print("samples.shape[0]=0 ", samples.shape[0])
            return 0
        acc=acc/samples.shape[0]
        return acc
    
    def export_info(self,path,evalvects,maxacc):
        confusion_matrix=np.empty((7,7))
        for i in range(evalvects.shape[0]):
            out=self._sess.run(self.softmax,
                               feed_dict={self.inputvect: evalvects[i][:170],
                                          self._iter_input: self._n_iterations,
                                           self.trainable: False})
            confusion_matrix[int(evalvects[i][170]-1),int(np.argmax(out))]+=1
        accvect=np.empty((1,7))
        accvect[0]=maxacc
        confusion_matrix=np.vstack((confusion_matrix,accvect))
        np.savetxt(path+".csv", confusion_matrix, delimiter=",")
            
    def train_nn(self, samples,exp_path,afew=None):
        
        print(afew)
        
        samplelen=samples.shape[0]
        evalidx=int(samplelen*0.2)
        evalvects=samples[evalidx:]
        
        samples=samples[:evalidx]
        maxacc=0
        metric_loss = 1000
        for iter_no in range(self._n_iterations):
            #Train with each vector one by one
            for i in range(len(samples)):
                none,metric_loss=self._sess.run([self.train,self.loss],
                               feed_dict={self.inputvect: samples[i][:170],
                                          self._iter_input: iter_no,
                                          self.y: get_label_vect(int(samples[i][170])), 
                                          self.trainable: True})
            acc=self.evaluate(evalvects,iter_no)
            afew_acc = 0.00
            if (afew is not None) & (len(afew)>0):
                afew_acc=self.evaluate(afew,iter_no)
            if acc>maxacc:
                maxacc=acc
            if acc>0.99:
                break
            if (afew is not None):
                if (len(afew)>0):
                    if (abs(afew[0])>0):
                        print("Epoch:", '%04d' % (iter_no+1),"cost=", "{:.9f}".format(metric_loss),",accuracy={:.6f}".format(acc),",afew_acc="+str(afew_acc))
                    else:
                        print("Epoch:", '%04d' % (iter_no+1),"cost=", "{:.9f}".format(metric_loss),",accuracy={:.6f}".format(acc))    
                else:
                    print("Epoch:", '%04d' % (iter_no+1),"cost=", "{:.9f}".format(metric_loss),
                  ",accuracy={:.6f}".format(acc))
            else:
                print("Epoch:", '%04d' % (iter_no+1),"cost=", "{:.9f}".format(metric_loss),
                  ",accuracy={:.6f}".format(acc))
        self.export_info(exp_path,evalvects,maxacc)
        
    def train_som(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """
        
        #Training iterations
        for iter_no in range(self._n_iterations):
            #Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._som_w_upd,
                               feed_dict={self.inputvect: input_vect,
                                          self._iter_input: iter_no})
        #Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid
 
        self._trained = True
 
    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid
 
    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """
 
        if not self._trained:
            raise ValueError("SOM not trained yet")
 
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect-
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])
 
        return to_return
