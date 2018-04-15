from FrameIterator_test import test_frame_iterator
from FrameProcessing import frame_processor
import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import data, exposure

#import scipy
#import matplotlib.pyplot as plt

def HoGFeatures(FRAMES_PATH, PREDICTOR_PATH):
    file_names = test_frame_iterator(FRAMES_PATH)
    fp = frame_processor(PREDICTOR_PATH)
    landmarks_num = [48, 45, 54, 57, 19, 24, 17, 21, 22, 26, 51, 36, 39, 37,
                     38, 41, 40, 42, 45, 43, 44, 47, 46, 33, 0, 16]
    cell_ = (32, 32)
    block_ = (4,4)
    HOGFeatures_collection = []
    HOGF_vect = ()
    for file_name in file_names:
        if not ".csv" in file_name:
            print('processing frame: ' + file_name)
            frame = cv2.imread(FRAMES_PATH + file_name)
            frame = cv2.resize(frame,(800,800))
            #landmarks = fp.get_all_landmarks(frame)
            selected_landmarks = fp.selected_landmarks(landmarks_num, frame)
            if selected_landmarks == None:
                continue
            if selected_landmarks == None:
                continue
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
            
            R_lip =  cv2.resize(R_lip,(200,130))#93, 203
            R_nose = cv2.resize(R_nose,(190,160))#159, 193
            R_leye = cv2.resize(R_leye,(200,130))#131, 214
            R_reye = cv2.resize(R_reye,(200,130))#
    
    

            #Potentionally error - cv2 works with BGR system when here is RGB is used
            
            R_leye_gray = cv2.cvtColor(R_leye, cv2.COLOR_BGR2GRAY)
            #R_leye_lbp = local_binary_pattern(R_leye_gray, 8, 1, method="nri_uniform")
            
            
            fd, hog_image = hog(R_leye_gray, orientations=9, pixels_per_cell=cell_,
                    cells_per_block=block_, visualise=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            
            ax1.axis('off')
            ax1.imshow(R_leye, cmap=plt.cm.gray)
            ax1.set_title('Input image')
            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            
            ax2.axis('off')
            ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            ax2.set_title('Histogram of Oriented Gradients')
            plt.show()
            
            
            R_leye_HoG = fd
            R_leye_hist = R_leye_HoG.ravel()
            R_leye_features = R_leye_hist.astype("float")
    
            R_reye_gray = cv2.cvtColor(R_reye, cv2.COLOR_BGR2GRAY)
            
            fd, hog_image = hog(R_reye_gray, orientations=9, pixels_per_cell=cell_,
                    cells_per_block=block_, visualise=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            
            ax1.axis('off')
            ax1.imshow(R_reye, cmap=plt.cm.gray)
            ax1.set_title('Input image')
            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            
            ax2.axis('off')
            ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            ax2.set_title('Histogram of Oriented Gradients')
            plt.show()
            
            R_reye_HoG = fd#_rescaled
            R_reye_hist = R_reye_HoG.ravel()
            R_reye_features = R_reye_hist.astype("float")
            '''
            R_reye_lbp = local_binary_pattern(R_reye_gray, 8, 1, method="nri_uniform")
            (R_reye_hist, _) = np.histogram(R_reye_lbp.ravel(), bins=np.arange(0, 60))
            R_reye_features = R_reye_hist.astype("float")
    '''
            R_nose_gray = cv2.cvtColor(R_nose, cv2.COLOR_BGR2GRAY)
            '''R_nose_lbp = local_binary_pattern(R_nose_gray, 8, 1, method="nri_uniform")
            (R_nose_hist, _) = np.histogram(R_nose_lbp.ravel(), bins=np.arange(0, 60))
            R_nose_features = R_nose_hist.astype("float")'''
            
            fd, hog_image = hog(R_nose_gray, orientations=9, pixels_per_cell=cell_,
                    cells_per_block=block_, visualise=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            
            ax1.axis('off')
            ax1.imshow(R_nose, cmap=plt.cm.gray)
            ax1.set_title('Input image')
            # Rescale histogram for better display
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            
            ax2.axis('off')
            ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            ax2.set_title('Histogram of Oriented Gradients')
            plt.show()
            
            R_nose_HoG = fd#_rescaled
            R_nose_hist = R_nose_HoG.ravel()
            R_nose_features = R_nose_hist.astype("float")
            
    
            R_lip_gray = cv2.cvtColor(R_lip, cv2.COLOR_BGR2GRAY)
            '''R_lip_lbp = local_binary_pattern(R_lip_gray, 8, 1, method="nri_uniform")
            (R_lip_hist, _) = np.histogram(R_lip_lbp.ravel(), bins=np.arange(0, 60))
            R_lip_features = R_lip_hist.astype("float")'''
            
            fd, hog_image = hog(R_lip_gray, orientations=9, pixels_per_cell=cell_,
                    cells_per_block=block_, visualise=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            
            ax1.axis('off')
            ax1.imshow(R_lip, cmap=plt.cm.gray)
            ax1.set_title('Input image')
            # Rescale histogram for better display
            #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            
            ax2.axis('off')
            ax2.imshow(hog_image, cmap=plt.cm.gray)
            ax2.set_title('Histogram of Oriented Gradients')
            plt.show()
            
            R_lip_HoG = fd
            R_lip_hist = R_lip_HoG.ravel()
            R_lip_features = R_lip_hist.astype("float")
    
            '''print(max(R_leye_features))
            (R_leye_features, _) = np.histogram(R_leye_features, bins = np.arange(0,0.5,0.01))
            (R_reye_features, _) = np.histogram(R_reye_features, bins = np.arange(0,0.5,0.01))
            (R_nose_features, _) = np.histogram(R_nose_features, bins = np.arange(0,0.5,0.01))
            (R_lip_features, _)  = np.histogram(R_lip_features, bins = np.arange(0,0.5,0.01))'''
    
            HOGFeatures_frame = np.concatenate((R_leye_features, R_reye_features,
                                                R_nose_features, R_lip_features), axis=0)
            print(len(HOGFeatures_frame))
            HOGFeatures_collection.append(HOGFeatures_frame)
    
    HOGF_vect = HOGFeatures_collection
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

    return HOGF_vect