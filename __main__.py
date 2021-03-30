#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 02:10:39 2021

@author: sk34nk
"""
import sys
import cv2
from car_detection import VehicheDetection_UsingMobileNetSSD as dv

def play_video(cap):
    ret, frame = cap.read()
    while(1):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
            cap.release()
            cv2.destroyAllWindows()
            break
        cv2.imshow('frame',frame)


def main(argv):
    filename = "resourse/video2.mp4"
    VC = cv2.VideoCapture(filename)
    if VC.isOpened(): 
        print("open")
    else:
        print("Not open")
    
    #play_video(VC)
    
    dv(filename)  

if __name__ == '__main__':
    main(sys.argv)
