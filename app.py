#================================================================ 
#Import necessary libraries
from flask import Flask, render_template, Response, request
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import pyttsx3
import time
from werkzeug.utils import secure_filename
from camera import startapplication
from camera2 import startapplication2

# Fall detection
import os,cv2
from PIL import Image as im
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import  torch
from PIL import Image
import imageio
import numpy as  np
from tensorflow.keras.preprocessing.image import img_to_array,load_img


UPLOAD_FOLDER = 'static/video_upload/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'sdkjfhsjkdfhskjdfhkjshdfkjshdf'
        
@app.route('/')
def Shady():
    return render_template('Shady.html')
    
@app.route('/FallDetect', methods=['GET', 'POST'])
def FallDetect():
    if request.method == "POST":
        f2= request.files['file1']
        print(f2)
        filename_secure = secure_filename(f2.filename)
        f2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_secure))
        print("print saved")
        filename1 = filename_secure
        print(filename1)

        vid_name = "static/video_upload/"+filename1

        vid = cv2.VideoCapture(vid_name)
        model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5/runs/train/exp/weights/best.pt',force_reload=True)
        color = np.random.uniform(0, 255, size=(3, 3))
        font = cv2.FONT_HERSHEY_PLAIN
        while(True):
            NumberofpersonInFrame=0  
            # Capture the video frame
            # by frame
            ret, frame = vid.read()
        
            # Display the resulting frame
            #cv2.imshow('frame', frame)
            if frame is None:
                continue
            img =frame# cv2.imread("room_ser.jpg")
            #img = cv2.resize(img, None, fx=0.4, fy=0.4)
            height, width, channels = img.shape

            results = model(frame)
            b=results.pandas().xyxy[0]
        #     print(type(b))
            name = b['name'].tolist()
            print("Predicted Output: ",name)
            xmin = b['xmin'].tolist()
            ymin = b['ymin'].tolist()
            xmax = b['xmax'].tolist()
            ymax = b['ymax'].tolist()   
            #if name=="walking":
            #    print("walking")
            #elif name=="sitting":
             #   print("sitting")
            #elif name=="waiting":
            #else:
            #    engine = pyttsx3.init()
            #    engine.say('Fall Detected')
             #   engine.runAndWait() 
            
        #     cv2.rectangle(img, (x, y), (int(xmin[0]), int(ymin[0])), color, 2)
            cv2.rectangle(img, (int(xmin[0]),int(ymin[0])), (int(xmax[0]),int(ymax[0])), (0,255,0), 2)

            cv2.putText(img, "prediction: "+str(name[0]), (30, 30), font, 1,(0,255,0), 3)
            cv2.imshow("Image", img)
            #time.sleep(5)
            
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()

    return render_template('FallDetection.html')

# @app.route('/liveFallDetection', methods=['GET', 'POST'])
# def LiveFallDetection():
#     if request.method == "POST":
#         user_inp = request.form.get("abc")
#         print("user_inp")
#         print(user_inp)

#         vid = cv2.VideoCapture(0)
#         model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5/runs/train/exp/weights/best.pt',force_reload=True)
#         color = np.random.uniform(0, 255, size=(3, 3))
#         font = cv2.FONT_HERSHEY_PLAIN
#         while(True):
#             NumberofpersonInFrame=0  
#             # Capture the video frame
#             # by frame
#             ret, frame = vid.read()
        
#             # Display the resulting frame
#             #cv2.imshow('frame', frame)
#             if frame is None:
#                 continue
#             img =frame# cv2.imread("room_ser.jpg")
#             #img = cv2.resize(img, None, fx=0.4, fy=0.4)
#             height, width, channels = img.shape

#             results = model(frame)
#             b=results.pandas().xyxy[0]
#         #     print(type(b))
#             name = b['name'].tolist()
#             print("Predicted Output: ",name)
#             xmin = b['xmin'].tolist()
#             ymin = b['ymin'].tolist()
#             xmax = b['xmax'].tolist()
#             ymax = b['ymax'].tolist()    
            
#         #     cv2.rectangle(img, (x, y), (int(xmin[0]), int(ymin[0])), color, 2)
#             cv2.rectangle(img, (int(xmin[0]),int(ymin[0])), (int(xmax[0]),int(ymax[0])), (0,255,0), 2)

#             cv2.putText(img, "prediction: "+str(name[0]), (30, 30), font, 1,(0,255,0), 3)
#             cv2.imshow("Image", img)
#             #time.sleep(5)
            
#             # the 'q' button is set as the
#             # quitting button you may use any
#             # desired button of your choice
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
        
#         # After the loop release the cap object
#         vid.release()
#         # Destroy all the windows
#         cv2.destroyAllWindows()

#     return render_template('FallDetection.html')

@app.route('/ObjectDetection', methods=['GET', 'POST'])
def ObjectDetection():
    global case
    case = 'object'
    return render_template('ObjectDetection.html')
    
@app.route('/SocialDistancingDetection', methods=['GET', 'POST'])
def SocialDistancingDetection():
    if request.method == "POST":
        user_inp = request.form.get("abc")
        print("user_inp")
        print(user_inp)

        fitToEllipse = False
        cap = cv2.VideoCapture(0)
        time.sleep(2)

        fgbg = cv2.createBackgroundSubtractorMOG2()
        j = 0

        while(1):
            ret, frame = cap.read()
            
            #Convert each frame to gray scale and subtract the background
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fgmask = fgbg.apply(gray)
                
                #Find contours
                contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                
                    # List to hold all areas
                    areas = []

                    for contour in contours:
                        ar = cv2.contourArea(contour)
                        areas.append(ar)
                    
                    max_area = max(areas, default = 0)

                    max_area_index = areas.index(max_area)

                    cnt = contours[max_area_index]

                    M = cv2.moments(cnt)
                    
                    x, y, w, h = cv2.boundingRect(cnt)

                    cv2.drawContours(fgmask, [cnt], 0, (255,255,255), 3, maxLevel = 0)
                    
                    if h < w:
                        j += 1
                        
                    if j > 10:
                        print("FALL")
                        cv2.putText(frame, 'FALL', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 2)
        #                 cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

                    if h > w:
                        j = 0 
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


                    cv2.imshow('video', frame)
                
                    if cv2.waitKey(33) == 27:
                        break
            except Exception as e:
                break
        cv2.destroyAllWindows()
    return render_template('SocialDistancingDetection.html')
    
@app.route('/VehicleCrashDetection', methods=['GET', 'POST'])
def VehicleCrashDetection():
    if request.method == "POST":
        f2= request.files['file1']
        print(f2)
        filename_secure = secure_filename(f2.filename)
        f2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_secure))
        print("print saved")
        filename1 = filename_secure
        print(filename1)
        vid = "static/video_upload/"+filename1
        startapplication(vid)

    return render_template('VehicleCrashDetection.html')

    
@app.route('/liveVehicleCrashDetection', methods=['GET', 'POST'])
def liveVehicleCrashDetection():
    if request.method == "POST":
        user_inp = request.form.get("abc")
        print("user_inp")
        print(user_inp)

        startapplication2()

    return render_template('liveVehicleCrashDetection.html')

@app.route('/ContactUs')
def ContactUs():
    return render_template('ContactUs.html')
    
@app.route('/Video', methods=['GET', 'POST'])
def Video():
    global video_link
    video_link = request.form.get('videolink')
    return render_template('Video.html')

if __name__ == "__main__":
    # app.run(debug=True)
    app.run("0.0.0.0")