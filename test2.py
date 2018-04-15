from os import walk
import av
import os
import dlib
import numpy as np
import cv2


#extract and save frames into directory with videosource
def extract_frames(directory):
    for subdirs, dirss, files in walk(directory):
           for fil in files:
               if ".avi" in fil:
                print(subdirs+"/"+fil)
                container = av.open(subdirs+"/"+fil)
            
                for i, frame in enumerate(container.decode(video=0)):
                    frame.to_image().save(subdirs+"/" +fil[:8]+"frame-%04d.jpg" % i)
                os.remove(subdirs+"/"+fil)

extract_frames("pics/mmi")
extract_frames("pics/AFEW")
extract_frames("pics/jaffe")
extract_frames("pics/ck")
extract_frames("pics/Stirling")

#Check, is there any face on the frame. If it is not, delete frame
def filter_frames(path):  
    detector = dlib. get_frontal_face_detector()
    count=0
    delcount=0
    for subdir, dirs, files in walk(path):
        for fil in files:
            if os.path.splitext(fil)[1].lower() in ('.jpg','.JPG','.jpeg','.tiff','.PNG','.png'):
                count+=1
                print(str(count)+" "+subdir+'/'+fil)
                frame = cv2.imread(subdir+'/'+fil)
                frame = cv2.resize(frame,(800,800))
                rects = detector(frame,1)
                if len(rects)==0:
                    delcount+=1
                    os.remove(subdir+'/'+fil)
        print("Deleted"+str(delcount))
    
filter_frames("pics/Stirling")
filter_frames("pics/mmi")
filter_frames("pics/jaffe")
filter_frames("pics/AFEW")
filter_frames("pics/ck")