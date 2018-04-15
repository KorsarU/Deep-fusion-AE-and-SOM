from FrameIterator_test import test_frame_iterator
from FrameProcessing import frame_processor
import cv2
import numpy as np

# Extracting facial geometric features as a displacement between the reference frame and every consequent frame

def GFeatures(FRAMES_PATH, PREDICTOR_PATH):
    file_names = test_frame_iterator(FRAMES_PATH)
    fp = frame_processor(PREDICTOR_PATH)
    landmarks_num = [48, 45, 54, 57, 19, 24, 17, 21, 22, 26, 51, 36, 39, 37,
                     38, 41, 40, 42, 45, 43, 44, 47, 46]
    #ref_coords = []
    GFeatures_collection = []
    GFvect = ()
    for file_name in file_names:
        if not ".csv" in file_name:
            print('processing frame: ' + file_name)
            frame = cv2.imread(FRAMES_PATH + file_name)
            frame = cv2.resize(frame,(800,800))
            #landmarks = fp.get_all_landmarks(frame)
            selected_landmarks = fp.selected_landmarks(landmarks_num, frame)
            current_coords = []
    
            for i in selected_landmarks:
                if i[0] in [17, 19, 21, 22, 24, 26, 51, 54, 57, 48]:
                    current_coords.append(i[1][0, 0])
                    current_coords.append(i[1][0, 1])
                    continue
            
            GFeatures_frame = np.array(current_coords)
            
            GFeatures_collection.append(GFeatures_frame)
            GFvect= GFeatures_collection

    return GFvect