#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 01:11:09 2021

@author: sk34nk
"""
import time
import cv2 as cv
import numpy as np
import math

# load our serialized model from disk
print("Load MobileNetSSD model")

prototxt_path = "MobileNetSSD_deploy.prototxt.txt"
model_path = "MobileNetSSD_deploy.caffemodel"

# initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

net = cv.dnn.readNetFromCaffe(prototxt_path, model_path)

def process_frame_MobileNetSSD(next_frame):
    rgb = cv.cvtColor(next_frame, cv.COLOR_BGR2RGB)
    (H, W) = next_frame.shape[:2]

    # convert the frame to a blob and pass the blob through the
    # network and obtain the detections
    blob = cv.dnn.blobFromImage(next_frame, size=(1280, 720), ddepth=cv.CV_8U)
    net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5, 127.5, 127.5])
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated
        # with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence`
        # is greater than the minimum confidence
        if confidence > 0.7:
            # extract the index of the class label from the
            # detections list
            idx = int(detections[0, 0, i, 1])
            # if the class label is not a car, ignore it
            if CLASSES[idx] not in ["motorbike", "car", "bicycle, bus"]:
                continue
            # compute the (x, y)-coordinates of the bounding box
            # for the object
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            
            cv.rectangle(next_frame, (startX, startY), (endX, endY), (0, 255, 0), 3)

    return next_frame

def VehicheDetection_UsingMobileNetSSD(filename):
    cap = cv.VideoCapture(filename)
    # Write output file
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fps = 20
    size = (int(frame_width),int(frame_height))
    fourcc = cv.VideoWriter_fourcc('m','p','4','v')
    out = cv.VideoWriter()
    success = out.open('output_mobilenetssd2.mov', fourcc, fps, size, True)

    frame_count = 0

    # start timer
    t1 = time.time()

    while True:
        ret, next_frame = cap.read() # Reads the next video frame into memory
        
        if ret == False: break

        frame_count += 1
        next_frame = process_frame_MobileNetSSD(next_frame)
        
        # write frame
        out.write(next_frame)
        
        key = cv.waitKey(50)
        
        if key == 27: # Hit ESC key to stop
            break
    
    
    # end timer
    t2 = time.time()

    # calculate FPS
    fps = str( float(frame_count / float(t2 - t1))) + ' FPS'

    print("/MobileNetSSD Car Detector")
    print("Frames processed: {}".format(frame_count))
    print("Elapsed time: {:.2f}".format(float(t2 - t1)))
    print("FPS: {}".format(fps))

    cap.release()
    cv.destroyAllWindows()
    out.release()