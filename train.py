from D_learn import SOM
from os import walk
import numpy as np
import pandas as pd


def train_eval(samples,path):
    print("Feeding hidden representation of fused features to SOM classifier")
    print("Initializing SOM classifier")
    som=SOM(10,8,170,400,learning_rate=0.01)
    #som = SOM(8, 5, 170, 400,learning_rate=0.007)
    print("Training SOM classifier")
    afew = mixemsamples(get_samples_from(path)[:1000])
    som.train_nn(samples,path,afew)
    print("Finished")
    som._sess.close()
    return som
    
def get_samples_from(path):
    count=-1
    samples=np.empty((1,171))
    for subdir, dirss, files in walk(path):
       if count==-1:
           dirs=dirss
           count+=1
       else:
           label=0
           if dirs[count]=="Anger":
               label=1
           elif dirs[count]=="Disgust":
               label=2
           elif dirs[count]=="Fear":
               label=3
           elif dirs[count]=="Sad":
               label=4
           elif dirs[count]=="Neutral":
               label=5
           elif dirs[count]=="Surprise":
               label=6
           elif dirs[count]=="Happy":
               label=7

           samplesb=pd.read_csv(path+dirs[count]+"/"+dirs[count]+"FusedHiddenRP.csv", sep=',',header=None)
           labeled= np.hstack((samplesb,np.full((samplesb.shape[0],1),label)))
           samples=np.vstack((samples,labeled))
           count+=1
           
    samples=np.delete(samples,0,0)
    
    print(samples.shape)
    if path=="pics/AFEW/":
        samples=mixemsamples(samples)[:6000]
    return samples

def mixemsamples(samples):
    samplesbuf=samples
    mixedsamples=np.empty((1,171))
    while samplesbuf.shape[0]>0 :
        idx = np.random.randint(0,samplesbuf.shape[0])
        mixedsamples = np.vstack((mixedsamples,samplesbuf[idx]))
        samplesbuf = np.delete(samplesbuf,idx,0)
        
    mixedsamples=np.delete(mixedsamples,0,0)
    print(mixedsamples)
    return mixedsamples
    
def get_label_vect(num):
    vect= np.zeros((7))
    vect[num-1]=1
    return vect

def get_num_from_vect(vect):
    for i in range(len(vect)):
        if vect[i]==1:
            return i+1

def automate(datasetfolders,name):
    sampols = np.empty((1,171))
    for folder in datasetfolders:
      if folder=="pics/AFEW":
        sampols=np.vstack((sampols,get_samples_from(folder)[1000:]))
      else:  
        sampols=np.vstack((sampols,get_samples_from(folder)))
    sampols=mixemsamples(sampols)
    sampols=np.delete(sampols,0,0)
    print(name)
    return train_eval(sampols,"pics/results/"+name)


soms = list()
for i in range(10):
    
    soms.append(("Stirling"+str(i), automate(["pics/Stirling/"],"Stirling"+str(i))))
    print("check")
    soms.append(("jaffe"+str(i), automate(["pics/jaffe/"],"jaffe"+str(i))))
    soms.append(("mmi"+str(i), automate(["pics/mmi/"],"mmi"+str(i))))
    soms.append(("ck"+str(i), automate(["pics/ck/"],"ck"+str(i))))
    soms.append(("Stirling"+str(i), automate(["pics/Stirling/"],"Stirling"+str(i))))
    #automate(["pics/AFEW/"],"AFEW"+str(i))
    soms.append(("ConstrainedMix"+str(i), automate(["pics/mmi/","pics/ck/","pics/jaffe/","pics/Stirling/"],"ConstrainedMix"+str(i))))
    soms.append(("SUPERMix"+str(i), automate(["pics/mmi/","pics/ck/","pics/jaffe/","pics/Stirling/"],"SUPERMix"+str(i))))
#,"pics/AFEW/"
    
index = 0;
for s in soms:
    print(s, " ", index)
    index = index + 1
print("check")
    
    
