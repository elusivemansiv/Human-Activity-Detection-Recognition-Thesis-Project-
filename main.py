import csv
import os
import numpy as np
import cv2
import mediapipe as mp

# Function to create a new CSV file or append to an existing one
def create_or_append_csv(filename, headers, data):
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(headers)
    with open(filename, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(data)

# Initialize variables
num_coords = 0
landmarks = []
class_name = ""

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detections
        results = holistic.process(image)

        if results.pose_landmarks and results.face_landmarks:
            num_coords = len(results.pose_landmarks.landmark) + len(results.face_landmarks.landmark)

        if num_coords > 0:
            landmarks = ['class']
            for val in range(1, num_coords + 1):
                landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        # Draw right hand landmarks
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

        # Draw left hand landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Export coordinates
        try:
            if class_name != "":
                # Extract pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                          for landmark in pose]).flatten())

                # Extract face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                          for landmark in face]).flatten())

                # Concatenate rows
                row = pose_row + face_row

                # Append class name
                row.insert(0, class_name)

                # Export to CSV
                create_or_append_csv('coords.csv', landmarks, row)

        except Exception as e:
            print("Error:", str(e))

        # Display webcam feed
        cv2.imshow('Raw Webcam Feed', image)

        # Wait for key press and change class name
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('h'):
            class_name = "Happy"
        elif key == ord('s'):
            class_name = "Sad"
        elif key == ord('v'):
            class_name = "Victorious"
        elif key == ord('t'):
            class_name = "Tired"
        elif key == ord('a'):
            class_name = "Angry"
        elif key == ord('d'):
            class_name = "Deprasion"
        elif key == ord('x'):
            class_name = "Suspicious"
        elif key == ord('f'):
            class_name = "Joking"

cap.release()
cv2.destroyAllWindows()