from tkinter import *
from PIL import Image, ImageTk
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np
import PIL.Image, PIL.ImageTk
import os
from pygame import mixer

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 5
COUNTER = 0
TOTAL = 0

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def start_detection():
    global vs, MYEPfrontend_root, panel, B1, detector, predictor, drowsy_label
    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=1).start()
    time.sleep(2.0)

    def update():
        global panel, B1, COUNTER, TOTAL
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < EYE_AR_THRESH:
             COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    if TOTAL >= 3:
                        drowsy_label.config(text="Drowsiness detected!", wraplength=150)  # Adjust wrap length as needed
                            # Generate sharper alarm
                        play_alarm()
                COUNTER = 0
            #cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        if panel is None:
            panel = Label(MYEPfrontend_root, image=frame)
            panel.image = frame
            panel.pack(side="left", padx=20, pady=10)
        else:
            panel.configure(image=frame)
            panel.image = frame
        MYEPfrontend_root.after(10, update)

    update()
    B1.config(state=DISABLED)

def play_alarm():
    mixer.init()
    mixer.music.load(os.path.join("D:\R.H.S\Final Year\B.E. Project\Final\MYEP_Sending", "danger_alert_siren.mp3"))
    mixer.music.play(loops=-1)  # Play the alarm sound indefinitely

MYEPfrontend_root = Tk()
MYEPfrontend_root.title("Alertdrive")
MYEPfrontend_root.geometry("720x480")  # Increased window width to accommodate message
MYEPfrontend_root.minsize(720, 480)
MYEPfrontend_root.resizable(False, False)  # Disable maximizing

background_image = Image.open("D:\R.H.S\Final Year\B.E. Project\Final\MYEP_Sending\\Surface Pro 8 - 3 (1).png")
background_image = background_image.resize((720, 480))
background_image = ImageTk.PhotoImage(background_image)

background_label = Label(MYEPfrontend_root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


frame = Frame(MYEPfrontend_root, borderwidth=1, bg="grey", relief=SUNKEN)
frame.place(x=600, y=412)

B1 = Button(frame, fg="Red", text="Initialise", command=start_detection)
B1.pack()

panel = None

drowsy_label = Label(MYEPfrontend_root, text="", fg="Red", font=("timesnewroman", 12, "bold"))
drowsy_label.place(x=590, y=280)  # Adjust the position of the label as per your preference

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

MYEPfrontend_root.mainloop()
