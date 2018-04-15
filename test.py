from D_learn import SOM
import tensorflow as tf
import numpy as np


"""
SOM intialization:
__init__(self, m, n, dim, n_iterations=100, alpha0=None,alphaf=None, sigma0=None, sigmaf=None, learning_rate=0.1):
- m, n: dimensions of SOM 
- 'n_iterations': number of iterations during training
- 'dim': dimensionality of the training data 
- 'alpha0,alphaf': initial and nth iteration values of alpha
- 'sigma0,sigma0': is the the initial and nth iteration neighbourhood value, denoting
           the radius of influence of the BMU while training. By default, its
           taken to be half of max(m, n).
"""
tf.device('/gpu:0')
som = SOM(10,8,4,3000,learning_rate=0.01)
writer = tf.summary.FileWriter("twriter/")

def test1():
    inputs = np.array([(2,1,2,3),(3,4,3,6),(4,7,2,1),(5,2,6,7)])
    labels = np.array([(0,1,0,0,0,0),(0,0,1,0,0,0),(0,0,0,0,1,0),(0,0,0,0,1,0)])
    
    som.train_nn(inputs,labels)

    #result=som._sess.run(som.sigcheck,feed_dict={som._vect_input:inputs[1],som._iter_input: 100})
    result=som._sess.run(som.softmax,feed_dict=
                         {som._vect_input:inputs[0],som._iter_input: som._n_iterations,som.trainable: False})
    
    
    with tf.Session():
        print(inputs[0])
        print(result)
        
    writer.add_graph(som._sess.graph)
    
def test2():
    vec1 = np.array([2,1,2,3])
    vec2 = np.array([3,4,3,6])
    
    result=tf.multiply(vec1,vec2)
    with tf.Session():
        print(result.eval())
        
test1()

