import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from mediapipe_plot_function import plot_from_csv

if __name__ == '__main__':
    wm = False
    format = 'portrait'
    quality = 1440

    # setting mediapipe parameters
    med_par = []
    # set static_image_mode (default: False); set to True if person detector is supposed to be done on every image (e.g. unrelated images instead of video); setting this to True leads to ignoring smooth_landmarks, smooth_segmentation and min_tracking_confidence
    med_par.append(False)
    # set model_complexity (default: 1, possible 0-2)
    med_par.append(1)
    # set smooth_landmarks (default: True); filters landmark positions; overruled by static image mode
    med_par.append(True)
    # set enable_segmentation (default: False); would also return segmentation mask additional to landmarks; Overruled, when entropy is given to process_video
    med_par.append(False)
    # set smooth_segmentation (default: True); filters segmentation mask; ignored when segmentation not enabled or static_image_mode=True
    med_par.append(True)
    # set min_detection_confidence (default: 0.5); Minimum confidence value from the person-detection model for the detection to be considered successful
    med_par.append(0.5)
    # set min_tracking_confidence (default: 0.5); Minimum confidence value from the landmark-tracking model for the pose landmarks to be considered tracked successfully
    # otherwise: person detection will be invoked on the next input image. Setting it to a higher value can increase robustness of the solution, at the expense of a higher latency
    med_par.append(0.4)

    with mp_pose.Pose(static_image_mode=med_par[0],
                    model_complexity=med_par[1],
                    smooth_landmarks=med_par[2],
                    enable_segmentation=med_par[3],
                    smooth_segmentation=med_par[4],
                    min_detection_confidence=med_par[5],
                    min_tracking_confidence=med_par[6],                   
                    ) as pose:

        print("-----------Starting Recording---------")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        width = int(cap.get(3))
        height = int(cap.get(4))
        scale = False
        if height < quality:
            scale = True
            if format=='landscape':
                width = int(quality/height * width)
                height = quality  
            else:
                height = int(quality/width * height)
                width = quality  
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame. Exiting.")
                break
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if scale:
                frame = cv2.resize(frame, (width,height), interpolation=cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            # Pose Estimation
            results = pose.process(frame)
            # Recolor back to BGR
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            w = width
            h = height
            x = 0
            y = 1

            if format == 'portrait':
                frame = cv2.flip(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), 1)
                w = height
                h = width
                x = 1
                y = 0
                # tmp = height
                # height = width
                # width = tmp

            if (results.pose_landmarks != None and not wm):
                xyz_list = []
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    xyz_list.append(landmark.x)
                    xyz_list.append(landmark.y)
                    xyz_list.append(landmark.z)
                    xyz_list.append(landmark.visibility)
                for i in mp_pose.POSE_CONNECTIONS:
                    start_point = tuple(np.multiply(np.array([xyz_list[i[0]*4+x],xyz_list[i[0]*4+y]]), [w, h]).astype(int))
                    end_point = tuple(np.multiply(np.array([xyz_list[i[1]*4+x],xyz_list[i[1]*4+y]]), [w, h]).astype(int))
                    thickness = int(max(min(w, h)/250,1))
                    cv2.line(frame, start_point, end_point, (200, 200, 200), thickness)
                for j in range(33):
                    drawing_coordinates = tuple(np.multiply(np.array([xyz_list[j*4+x],xyz_list[j*4+y]]), [w, h]).astype(int))
                    radius = int(max(min(w, h)/100,1))
                    blue_part = int(255*xyz_list[j*4+3])
                    green_part = int(255*(1-xyz_list[j*4+3]))
                    red_part = 0
                    cv2.circle(frame, drawing_coordinates, radius, (blue_part, green_part, red_part), -1) # green = (0, 100, 0)
            cv2.imshow('Demonstrator',frame)

            if cv2.waitKey(1) == ord('q'):
                break

        print("------------STOPPED---------------")  