from FrameIterator_test import test_frame_iterator
from FrameProcessing import frame_processor
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
#import scipy
#import matplotlib.pyplot as plt

def LBPFeatures(FRAMES_PATH, PREDICTOR_PATH):
    file_names = test_frame_iterator(FRAMES_PATH)
    fp = frame_processor(PREDICTOR_PATH)
    landmarks_num = [48, 45, 54, 57, 19, 24, 17, 21, 22, 26, 51, 36, 39, 37,
                     38, 41, 40, 42, 45, 43, 44, 47, 46, 33, 0, 16]
    LBPFeatures_collection = []
    LBPF_vect = ()
    for file_name in file_names:
        if not ".csv" in file_name:
            print('processing frame: ' + file_name)
            frame = cv2.imread(FRAMES_PATH + file_name)
            frame = cv2.resize(frame,(800,800))
            #landmarks = fp.get_all_landmarks(frame)
            selected_landmarks = fp.selected_landmarks(landmarks_num, frame)
            leye_top, leye_bottom, reye_top, reye_bottom, leye_center, reye_center = fp.eyes_landmarks_estimation(frame)
    
            for i in selected_landmarks:
                if i[0] == 51:
                    top_lip = (i[1][0, 0], i[1][0, 1])
                if i[0] == 33:
                    nose_tip = (i[1][0, 0], i[1][0, 1])
                if i[0] == 48:
                    lip_left_corner = (i[1][0, 0], i[1][0, 1])
                if i[0] == 54:
                    lip_right_corner = (i[1][0, 0], i[1][0, 1])
                if i[0] == 36:
                    eye_left_corner = (i[1][0, 0], i[1][0, 1])
                if i[0] == 45:
                    eye_right_corner = (i[1][0, 0], i[1][0, 1])
                if i[0] == 57:
                    lip_bottom = (i[1][0, 0], i[1][0, 1])
                if i[0] == 51:
                    lip_top = (i[1][0, 0], i[1][0, 1])
                if i[0] == 0:
                    face_left_side = (i[1][0, 0], i[1][0, 1])
                if i[0] == 16:
                    face_right_side = (i[1][0, 0], i[1][0, 1])
                if i[0] == 19:
                    left_brow_top = (i[1][0, 0], i[1][0, 1])
                if i[0] == 24:
                    right_brow_top = (i[1][0, 0], i[1][0, 1])
                if i[0] == 24:
                    right_brow_top = (i[1][0, 0], i[1][0, 1])
                if i[0] == 17:
                    left_brow_left_side = (i[1][0, 0], i[1][0, 1])
                if i[0] == 26:
                    right_brow_right_side = (i[1][0, 0], i[1][0, 1])
                if i[0] == 21:
                    left_brow_right_side = (i[1][0, 0], i[1][0, 1])
                if i[0] == 22:
                    right_brow_left_side = (i[1][0, 0], i[1][0, 1])
    
            H_nose = int(nose_tip[1] + (top_lip[1] - nose_tip[1]) / 2)
            lip_left_edge_x = lip_left_corner[0] + int((eye_left_corner[0] - lip_left_corner[0]) / 2)
            lip_right_edge_x = lip_right_corner[0] + int((eye_right_corner[0] - lip_right_corner[0]) / 2)
            leyex = left_brow_left_side[0] + int((face_left_side[0] - left_brow_left_side[0]) / 2)
            leyex_w = left_brow_right_side[0] + int((right_brow_left_side[0] - left_brow_right_side[0]) / 2)
            leyey = left_brow_top[1] - 10
            leyey_h = leye_bottom[1] + int((nose_tip[1] - leye_bottom[1]) / 3)
            reyex_w = right_brow_right_side[0] + int((face_right_side[0] - right_brow_right_side[0]) / 2)
            reyex = right_brow_left_side[0] - int((right_brow_left_side[0] - left_brow_right_side[0]) / 2)
            reyey = right_brow_top[1] - 10
            reyey_h = reye_bottom[1] + int((nose_tip[1] - leye_bottom[1]) / 3)
            R_lip = frame[H_nose: lip_bottom[1] + 2 * (lip_top[1] - H_nose), lip_left_edge_x: lip_right_edge_x]
            R_nose = frame[leye_bottom[1]:H_nose, leye_bottom[0]:reye_bottom[0]]
            R_leye = frame[leyey:leyey_h, leyex:leyex_w]
            R_reye = frame[reyey:reyey_h, reyex:reyex_w]
    
    
    
            #Potentionally error - cv2 works with BGR system when here is RGB is used
            
            R_leye_gray = cv2.cvtColor(R_leye, cv2.COLOR_RGB2GRAY)
            R_leye_lbp = local_binary_pattern(R_leye_gray, 8, 1, method="nri_uniform")
            (R_leye_hist, _) = np.histogram(R_leye_lbp.ravel(), bins=np.arange(0, 60))
            R_leye_features = R_leye_hist.astype("float")
    
            R_reye_gray = cv2.cvtColor(R_reye, cv2.COLOR_RGB2GRAY)
            R_reye_lbp = local_binary_pattern(R_reye_gray, 8, 1, method="nri_uniform")
            (R_reye_hist, _) = np.histogram(R_reye_lbp.ravel(), bins=np.arange(0, 60))
            R_reye_features = R_reye_hist.astype("float")
    
            R_nose_gray = cv2.cvtColor(R_nose, cv2.COLOR_RGB2GRAY)
            R_nose_lbp = local_binary_pattern(R_nose_gray, 8, 1, method="nri_uniform")
            (R_nose_hist, _) = np.histogram(R_nose_lbp.ravel(), bins=np.arange(0, 60))
            R_nose_features = R_nose_hist.astype("float")
    
            R_lip_gray = cv2.cvtColor(R_lip, cv2.COLOR_RGB2GRAY)
            R_lip_lbp = local_binary_pattern(R_lip_gray, 8, 1, method="nri_uniform")
            (R_lip_hist, _) = np.histogram(R_lip_lbp.ravel(), bins=np.arange(0, 60))
            R_lip_features = R_lip_hist.astype("float")
    
    
    
    
            LBPFeatures_frame = np.concatenate((R_leye_features, R_reye_features,
                                                R_nose_features, R_lip_features), axis=0)
    
            LBPFeatures_collection.append(LBPFeatures_frame)
    
            LBPF_vect = LBPFeatures_collection
                    # exclude the first frame in the sequence
    
            #plt.figure()
            #plt.subplot(1, 4, 1)
            #plt.imshow(R_lip)
            #plt.subplot(1, 4, 2)
            #plt.imshow(R_nose)
            #plt.subplot(1, 4, 3)
            #plt.imshow(R_leye)
           	#plt.subplot(1, 4, 4)
            #plt.imshow(R_reye)
            #plt.show()

    return LBPF_vect
