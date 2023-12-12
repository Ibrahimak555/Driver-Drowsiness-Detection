from scipy.spatial import distance as dist
from imutils import face_utils
from pygame import mixer
import numpy as np
import imutils
import dlib
import cv2

mixer.init()
mixer.music.load("music.wav")

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Change this Theshold value based on distance of camera from driver
# threshold 20 is optimal for about 30 cm
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

        distance = lip_distance(shape)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if (distance > YAWN_THRESH):
            YAWN_COUNTER += 1

            if YAWN_COUNTER >= YAWN_CONSEC_FRAMES:
                cv2.putText(frame, "***Yawn Alert***", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10,325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            YAWN_COUNTER = 0

        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.release()
