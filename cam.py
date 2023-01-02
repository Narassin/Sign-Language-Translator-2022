import cv2
import numpy as np
import time
import skimage

# import sys

from skimage.feature import greycomatrix, graycoprops

# ----------------- calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 ----------------------------------
def calc_glcm_all_agls(img, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    
    glcm = greycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in skimage.feature.graycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)
    
    return feature

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # h, w = gray.shape
    # ymin, ymax, xmin, xmax = h//3, h*2//3, w//3, w*2//3
    # crop = gray[ymin:ymax, xmin:xmax]
            
    resize = cv2.resize(gray,(1,24))
    # properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    # wrk = calc_glcm_all_agls(resize, props=properties)
    image_array = np.asarray(resize)
    
    image_array = np.reshape(image_array, (24,1))
    
    
    # for (x,y,w,h) in wrk:
    #     cv2.rectangle(frame,(x,y),(x+w, y+h), (0,255,0), 2)
    
    # time.sleep(3)
    cv2.imshow('VideoTest',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




video_capture.release()
cv2.destroyAllWindows