from FrameIterator_test import get_unique_paths, test_frame_iterator
from GFExtractor import GFeatures
from LBPFExtractor import LBPFeatures
from HoGExtractor import HoGFeatures
import numpy as np

def extractor(path,name, featuresType = [1,2,3]):
    unique_paths = ' '
    unique_paths = get_unique_paths(path)
    
    print(unique_paths)

   


    if 1 in featuresType:
        print('HoG features extraction ...')
    
        HOGF = []
        un = path+"/"+name+"/"
        files = test_frame_iterator(un)
        for file in files:
            print(un+file)
            
        HOG_ = HoGFeatures(un, 'face_landmarks/shape_predictor_68_face_landmarks.dat')
        for _ in HOG_:	
            HOGF.append(_)
    
        print(len(HOGF))
        np.savetxt(path+"/"+name+"/"+name+"HOGF.csv", HOGF, delimiter=",")

    if 2 in featuresType:    
        print('LBP features extraction ...')
    
        LBPF = []
        for un in unique_paths:
            files = test_frame_iterator(un)
            for file in files:
                print(un+file)
                
            LBPF_ = LBPFeatures(un, 'face_landmarks/shape_predictor_68_face_landmarks.dat')
            for _ in LBPF_:	
                LBPF.append(_)
    
        print(len(LBPF))
        np.savetxt(path+"/"+name+"LBPF.csv", LBPF, delimiter=",")
    if 3 in featuresType:
        print('Geometric features extraction ...')
    
        GF = []
        for un in unique_paths:
            files = test_frame_iterator(un)
            for file in files:
                print(un+file)
                GF_ = GFeatures(un, 'face_landmarks/shape_predictor_68_face_landmarks.dat')
                for _ in GF_:
                    GF.append(_)
    
                print(len(GF))
    
        np.savetxt(path+"/"+name+"GF.csv", GF, delimiter=",")

#extractor("pics/test","Angry", [1])