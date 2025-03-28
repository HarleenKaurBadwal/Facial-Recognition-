
import tkinter as tk
from tkinter import Message ,Text
import cv2
import os

import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
import os.path

from cv2 import WINDOW_NORMAL
from face_detection import find_faces

ESC = 27

window = tk.Tk()
#helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
window.title("Face_Recogniser")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
#answer = messagebox.askquestion(dialog_title, dialog_text)
 
#window.geometry('1280x720')
window.configure(background='green')

# from here start function


def start_webcam(model_emotion, model_gender, window_size, window_name='live', update_time=50):
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    video_feed = cv2.VideoCapture(0)
    video_feed.set(3, width)
    video_feed.set(4, height)
    read_value, webcam_image = video_feed.read()

    delay = 0
    init = True
    while read_value:
        read_value, webcam_image = video_feed.read()
        for normalized_face, (x, y, w, h) in find_faces(webcam_image):
          if init or delay == 0:
            init = False
            emotion_prediction = model_emotion.predict(normalized_face)
            gender_prediction = model_gender.predict(normalized_face)
          if (gender_prediction[0] == 0):
              cv2.rectangle(webcam_image, (x,y), (x+w, y+h), (0,0,255), 2)
          else:
              cv2.rectangle(webcam_image, (x,y), (x+w, y+h), (255,0,0), 2)
          cv2.putText(webcam_image, emotions[emotion_prediction[0]], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
        delay += 1
        delay %= 20
        cv2.imshow(window_name, webcam_image)
        key = cv2.waitKey(update_time)
        if key == ESC:
            break

    cv2.destroyWindow(window_name)

def analyze_picture(model_emotion, model_gender, path, window_size, window_name='static'):
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    image = cv2.imread(path, 1)
    for normalized_face, (x, y, w, h) in find_faces(image):
        emotion_prediction = model_emotion.predict(normalized_face)
        gender_prediction = model_gender.predict(normalized_face)
        if (gender_prediction[0] == 0):
            cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
        else:
            cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(image, emotions[emotion_prediction[0]], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow(window_name, image)
    key = cv2.waitKey(0)
    if key == ESC:
        cv2.destroyWindow(window_name)

if __name__ == '__main__':
    emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"]

    # Load model
    fisher_face_emotion = cv2.face.FisherFaceRecognizer_create()
    fisher_face_emotion.read('models/emotion_classifier_model.xml')

    fisher_face_gender = cv2.face.FisherFaceRecognizer_create()
    fisher_face_gender.read('models/gender_classifier_model.xml')

    # Use model to predict
    choice = input("Use webcam?(y/n) ")
    if (choice == 'y'):
        window_name = "Facifier Webcam (press ESC to exit)"
        start_webcam(fisher_face_emotion, fisher_face_gender, window_size=(1280, 720), window_name=window_name, update_time=15)
    elif (choice == 'n'):
        run_loop = True
        window_name = "Facifier Static (press ESC to exit)"
        print("Default path is set to data/sample/")
        print("Type q or quit to end program")
        while run_loop:
            path = "../data/sample/"
            file_name = input("Specify image file: ")
            if file_name == "q" or file_name == "quit":
                run_loop = False
            else:
                path += file_name
                if os.path.isfile(path):
                    analyze_picture(fisher_face_emotion, fisher_face_gender, path, window_size=(1280, 720), window_name=window_name)
                else:
                    print("File not found!")
    else:
        print("Invalid input, exiting program.")




#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message = tk.Label(window, text="Face-Recognition-Based-Attendance-Management-System" ,bg="blue"  ,fg="white"  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 

message.place(x=200, y=20)

quitWindow = tk.Button(window, text="TakeImage", command=start_webcam ,fg="red"  ,bg="yellow"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=500, y=500)

window.mainloop()
