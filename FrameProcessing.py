import cv2
import dlib
import numpy as np


#main frame processing class (detecting landmarks, selecting landmarks, detecting regions of interest,
# )
class frame_processor:
    # initializing frame processing class by passing in 68 landmarks prediction classifier
    def __init__(self, PREDICTOR_PATH ):
        self.PREDICTOR_PATH = PREDICTOR_PATH
        
    #Returns list of pairs landmark_idx, xy_tuple
    def get_all_landmarks(self, frame):
        self.frame = frame
        predictor = dlib.shape_predictor(self.PREDICTOR_PATH)
        detector = dlib.get_frontal_face_detector()
        rects = detector(frame,1)
        if len(rects) == 0:
            return None
        landmarks_coords = np.matrix([[p.x, p.y] for p in (predictor(frame, rects[0])).parts()])
        landmarks_coords = list(enumerate(landmarks_coords))
        return landmarks_coords

    #Returns frame with landmarks marked on it
    def anotate_all_landmarks(self,frame):
        self.frame = frame
        landmarks = self.get_all_landmarks(frame)
        frame_ = frame.copy()
        for landmark in landmarks:
            idx = landmark[0]
            posx = landmark[1][0, 0]
            posy = landmark[1][0, 1]
            pos = (posx, posy)
            cv2.putText(frame_, str(idx), pos, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, (0, 255, 255))
            cv2.circle(frame_, pos, 3, color=(0, 255, 0))
        return frame_

    #Extracts specific landmarks from fiven list
    def selected_landmarks(self, lms_idx, frame):
        self.lms_idx = lms_idx
        self.frame = frame
        sel_landmarks = list()
        landmarks = self.get_all_landmarks(frame)
        if landmarks == None:
            return landmarks
        for landmark in landmarks:
            if landmark[0] in lms_idx:
                sel_landmarks.append(landmark)
            else:
                pass
        return sel_landmarks

    #Calculates leye_top leye_bottom reye_top reye_bottom leye_center reye_center
    def eyes_landmarks_estimation(self, frame):
        self.frame = frame
        landmarks = self.get_all_landmarks(frame)
        x_coords = list()
        y_coords = list()
        #estimating left eye top coordinates
        for landmark in landmarks:
            if landmark[0] == 37 or landmark[0] == 38:
                x_coords.append(landmark[1][0,0])
                y_coords.append(landmark[1][0,1])
        x_leye_top = int((x_coords[0] + x_coords[1]) / 2)
        y_leye_top = int((y_coords[0] + y_coords[1]) / 2)
        leye_top = (x_leye_top, y_leye_top)
        x_coords = []
        y_coords = []
        #estimating left eye bottom coordinates
        for landmark in landmarks:
            if landmark[0] == 41 or landmark[0] == 40:
                x_coords.append(landmark[1][0,0])
                y_coords.append(landmark[1][0,1])
        x_leye_bottom = int((x_coords[0] + x_coords[1]) / 2)
        y_leye_bottom = int((y_coords[0] + y_coords[1]) / 2)
        leye_bottom = (x_leye_bottom, y_leye_bottom)
        x_coords = []
        y_coords = []
        #estimating left eye center coordinates
        leye_x_center = int((leye_top[0] + leye_bottom[0]) / 2)
        leye_y_center = leye_bottom[1] - int(abs((leye_top[1] - leye_bottom[1]) / 2))
        leye_center = (leye_x_center, leye_y_center)
        #estimating right eye top coordinates
        for landmark in landmarks:
            if landmark[0] == 43 or landmark[0] == 44:
                x_coords.append(landmark[1][0, 0])
                y_coords.append(landmark[1][0, 1])
        x_reye_top = int((x_coords[0] + x_coords[1]) / 2)
        y_reye_top = int((y_coords[0] + y_coords[1]) / 2)
        reye_top = (x_reye_top, y_reye_top)
        x_coords = []
        y_coords = []
        #estimating right eye bottom coordinates
        for landmark in landmarks:
            if landmark[0] == 47 or landmark[0] == 46:
                x_coords.append(landmark[1][0, 0])
                y_coords.append(landmark[1][0, 1])
        x_reye_bottom = int((x_coords[0] + x_coords[1]) / 2)
        y_reye_bottom = int((y_coords[0] + y_coords[1]) / 2)
        reye_bottom = (x_reye_bottom, y_reye_bottom)
        x_coords = []
        y_coords = []
        #estimating right eye center coordinates
        reye_x_center = int((reye_top[0] + reye_bottom[0]) /2)
        reye_y_center = reye_bottom[1] - int(abs((reye_top[1] - reye_bottom[1]) / 2))
        reye_center = (reye_x_center, reye_y_center)
        estimated_eye_landmarks = np.array([leye_top,
                                            leye_bottom,
                                            reye_top,
                                            reye_bottom,
                                            leye_center,
                                            reye_center])
        return estimated_eye_landmarks

    def draw_estimated_landmarks(self, frame, landmarks_coords_list):
        self.landmarks_coords_list = landmarks_coords_list
        self.frame = frame
        frame_ = frame.copy()
        for landmarks_coords in landmarks_coords_list:
            posx = landmarks_coords[0]
            posy = landmarks_coords[1]
            pos = (posx, posy)
            cv2.circle(frame_, pos, 3, color=(255, 255, 0))
        return frame_





