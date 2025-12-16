import cv2
import numpy as np

import imutils

import process
import os.path

markerDict = {}

def readMarker(markerId, markerType, markerResolution=None, debug=False, path="./"):
    if markerResolution == None:
        if (f"{markerType}_{markerId}" in markerDict):
            return markerDict[f"{markerType}_{markerId}"]        
        else:
            
            if markerType in ["36h11", "16h5"]:
                x1, x2 = markerType.split("h")
                file = f"{path}tag{x1}h{x2}/tag{x1}_%02d_%05d.png" % (int(x2), int(markerId))
            elif markerType in ['shape']:
                file = f"{path}{markerType}/{markerId}_8mm.png"
            else:
                file = f"{path}{markerId}.png"
            if debug:
                print(file)
            if (os.path.isfile(file)):
                marker = (255-cv2.imread(file)[:,:,0]) / 255.0

                markerDict[f"{markerType}_{markerId}"] = marker

                return marker
            else:
                print(f"File {file} does not exist")
    else:
        if (f"{markerType}_{markerId}_{markerResolution}" in markerDict):
            return markerDict[f"{markerType}_{markerId}_{markerResolution}"]
        else:
            

            if markerType in ["36h11", "16h5"]:
                x1, x2 = markerType.split("h")
                #x1, x2 = int(x1), int(x2)
                file = f"{path}tag{x1}h{x2}/tag{x1}_%02d_%05d.png" % (int(x2), int(markerId))
            else:
                file = f"{path}{markerType}/{markerId}_8mm.png"

            if debug:
                print(file)
                
            if (os.path.isfile(file)):
                marker = (255-cv2.imread(file)[:,:,0]) / 255.0
            else:
                print(f"File {file} does not exist")

            marker2 = np.zeros((markerResolution, markerResolution), dtype=np.float32)
            
            c = np.array([markerResolution, markerResolution])//2
            wh = np.array(marker.shape)//2
            x1, y1 = c-wh
            x2, y2 = c+wh
            marker2[x1:x2,y1:y2] = marker

            markerDict[f"{markerType}_{markerId}_{markerResolution}"] = marker2
            
            del file
            del marker
            
            return marker2

def load_tag(size, marker_type, markerId, angle, PixelSizeInMM = 4.022, MARKER_PATH = './'):
    
    AprilTagPixelSizeInMM = size
    template = readMarker(markerId, marker_type, markerResolution = None, path=f"{MARKER_PATH}")

    if marker_type in ['16h5', '36h11']:
        s = np.array(template.shape) / (PixelSizeInMM/AprilTagPixelSizeInMM)
        s = np.round(s * 4).astype(int)
        template = cv2.resize(template, dsize=tuple(s), interpolation=cv2.INTER_NEAREST)
        del s
    else:
        s = size/PixelSizeInMM
        s = np.round(s * 4).astype(int)
        #template = np.pad(template, pad_width = 10) # pad_width possibly needs adjustment
        template = cv2.resize(template, dsize=(s,s), interpolation=cv2.INTER_LANCZOS4)
        template = np.clip(template, 0, 255)
        del s
        #template = np.pad(template, pad_width = 1)
        
    templateRet = process.getNorm(imutils.rotate_bound(template, angle)).astype(np.float32)
    del template
    temp = np.array([128,128])
    w,h = temp - templateRet.shape

    if w < 0:
        c = np.array(templateRet.shape)//2
        wh = temp//2
        x1, y1 = c-wh
        x2, y2 = c+wh
        templateRet = templateRet[x1:x2,y1:y2]
    elif w > 0:
        templateRet = np.pad(templateRet, np.array([(np.ceil(w/2),w-np.ceil(w/2)),(np.ceil(h/2),h - np.ceil(h/2))]).astype(int))
    else:
        templateRet = templateRet

    del temp
    
    return templateRet