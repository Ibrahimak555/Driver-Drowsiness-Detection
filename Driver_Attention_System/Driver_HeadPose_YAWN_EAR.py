import cv2
import time
import numpy as np
import mediapipe as mp
from pygame import mixer

# Alert signal initialization
mixer.init()
mixer.music.load("music.wav")

# Loading Mediapipe facemesh containing 478 labeled points
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Drawing FaceMesh on screen
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
drawing_spec_2 = mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1)

# Head Pose
HeadPose_Frames = 25
HeadPose_Counter = 0

# EAR - Eye Aspect Ratio
EAR_Thresh = 0.25
EAR_Frames = 30
EAR_Counter = 0

# Yawn
YAWN_Thresh = 0.3
YAWN_Frames = 36
YAWN_Counter = 0

# Features for specified use
HeadPose_Features = [1,33,61,199,263,291]
LeftEye_Features = [263,362,386,374]
RightEye_Features = [33,133,145,159]
Mouth_Features = [13,14,78,308]

# Start capturing video
vid = cv2.VideoCapture(0)

while vid.isOpened():
    retval, image = vid.read()
    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False # To improve performance
    results = face_mesh.process(image) # Get the result
    image.flags.writeable = True # To improve performance

    image = cv2.cvtColor(image, cv2. COLOR_RGB2BGR) # Convert the color space from RGB to BGR

    img_h, img_w, img_c = image.shape
    face_3d = []; face_2d = []
    l_eye = []; r_eye = []
    mouth = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):

                # Head Pose Estimation
                if idx in HeadPose_Features:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])       # Get the 2D Coordinates
                    face_3d.append([x, y, lm.z]) # Get the 3D Coordinates

                # Left Eye
                if idx in LeftEye_Features:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    l_eye.append([x,y])

                # Right Eye
                if idx in RightEye_Features:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    r_eye.append([x,y])
                
                # Mouth
                if idx in Mouth_Features:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    mouth.append([x,y])

            # ---------- Head Pose Estimation ----------
                    
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w # The camera matrix
            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])
            
            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP (Pose n Position)
            retval, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head tilting
            distracted = True
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                distracted = False
                text = "Forward"

            if distracted:
                HeadPose_Counter += 1

                if HeadPose_Counter > HeadPose_Frames:
                    cv2.putText(image, "Distracted!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                HeadPose_Counter = 0
            
            # Display the nose direction
            
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,255,0), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (520, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(image, "y: " + str(np.round(y,2)), (520, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(image, "z: " + str(np.round(z,2)), (520, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # ---------- EAR ----------

            l_EAR_Width = abs(l_eye[0][0] - l_eye[1][0]) # Points 33, 133
            l_EAR_Height = abs(l_eye[2][1] - l_eye[3][1]) # Points 145, 159
            l_EAR = l_EAR_Height / l_EAR_Width

            r_EAR_Width = abs(r_eye[0][0] - r_eye[1][0]) # Points 263, 362
            r_EAR_Height = abs(r_eye[2][1] - r_eye[3][1]) # Points 386, 374
            r_EAR = r_EAR_Height / r_EAR_Width
            
            # Calculating EAR as average of both eyes
            EAR = (l_EAR + r_EAR) / 2.0

            cv2.putText(image, "EAR: {:.2f}".format(EAR), (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if EAR < EAR_Thresh:
                EAR_Counter += 1

                if EAR_Counter > EAR_Frames:
                    cv2.putText(image, "Eye Closed!", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                EAR_Counter = 0

            # ---------- YAWN ----------
                
            mouth_Height = abs(mouth[0][1] - mouth[1][1]) # Points 13, 14
            mouth_Width = abs(mouth[2][0] - mouth[3][0]) # Points 78, 308

            YAWN = mouth_Height / mouth_Width

            cv2.putText(image, "YAWN: {:.2f}".format(YAWN), (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if YAWN > YAWN_Thresh:
                YAWN_Counter += 1

                if YAWN_Counter > YAWN_Frames:
                    cv2.putText(image, "Yawning!", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                YAWN_Counter = 0
        
        # FPS calculation
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        cv2.putText(image, f'FPS: {int(fps)}', (500,470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = drawing_spec_2,
                    connection_drawing_spec = drawing_spec)
    
    cv2.imshow('Driver Attention System - Press \'esc\' to exit', image)

    if cv2.waitKey(5) & 0xFF == 27: 
        break

vid.release()