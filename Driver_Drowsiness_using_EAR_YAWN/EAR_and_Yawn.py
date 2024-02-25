from scipy.spatial import distance as dist
from imutils import face_utils
from pygame import mixer
import numpy as np
import imutils
import dlib
import cv2

mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# The distance of face from the camera must be between 30cm - 45cm
# The Eye closeness will be detected if closed for more than 30 frames (2.5 secs).
# The Yawn will be detected if mouth opened for more than 36 frames (3 secs).
# NOTE: Yawn threshold will change from person to person ranging between 17 - 25.
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 30
EYE_COUNTER = 0
YAWN_THRESH = 19
YAWN_CONSEC_FRAMES = 36
YAWN_COUNTER = 0

print("-> Loading the predictor and detector...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("-> Starting Video Stream")
vs = cv2.VideoCapture(0)

while True:

    ret, frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye =  eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull],  -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if (ear < EYE_AR_THRESH):
            EYE_COUNTER += 1
            
            if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Eye Closed!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if ((EYE_COUNTER-EYE_AR_CONSEC_FRAMES) % 70) == 0:
                    mixer.music.play()
        else:
            EYE_COUNTER = 0

        if (distance > YAWN_THRESH):
            YAWN_COUNTER += 1

            if YAWN_COUNTER >= YAWN_CONSEC_FRAMES:
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if ((YAWN_COUNTER-YAWN_CONSEC_FRAMES) % 70) == 0:
                    mixer.music.play()
        else:
            YAWN_COUNTER = 0

        # cv2.putText(frame, "Eye_counter:  "+str(EYE_COUNTER), (10, 60),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "Yawn_counter: "+str(YAWN_COUNTER), (10, 85),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"): # Press 'q' to close the window.
        break

cv2.destroyAllWindows()
vs.release()