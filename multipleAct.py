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
num_body_coords = 0
body_landmarks = []
class_name = ""

# Initialize video capture
cap = cv2.VideoCapture('videos2/smash-and-grab robbery.mp4')  # Replace 'your_video_file.mp4' with your video file path

# Initialize holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Video ended.")
            break

        # Recolor feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detections
        results = holistic.process(image)

        if results.pose_landmarks:
            num_body_coords = len(results.pose_landmarks.landmark)

        if num_body_coords > 0 and not body_landmarks:  # Check if 'body_landmarks' is empty before updating it
            body_landmarks = ['class']
            for val in range(1, num_body_coords + 1):
                body_landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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

                # Concatenate rows
                row = pose_row

                # Append class name
                row.insert(0, class_name)

                # Export to CSV
                create_or_append_csv('coords1.csv', body_landmarks, row)

        except Exception as e:
            print("Error:", str(e))

        # Display video feed
        cv2.imshow('Raw Video Feed', image)

        # Wait for key press and change class name
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('f'):
            class_name = "Frightened"
        elif key == ord('a'):
            class_name = "Action"
        elif key == ord('b'):
            class_name = "Breaking"
        elif key == ord('t'):
            class_name = "Trespassing"
        elif key == ord('s'):
            class_name = "Stealing"
        elif key == ord('x'):
            class_name = "Suspicious"
        elif key == ord('w'):
            class_name = "walking"
        elif key == ord('d'):
            class_name = "Falldown"
        elif key == ord('c'):
            class_name = "Climb"
        elif key == ord('z'):
            class_name = "Stand"
        elif key == ord('r'):
            class_name = "Running"
        elif key == ord('v'):
            class_name = "Cycling"
        elif key == ord('e'):
            class_name = "Sitting"
        elif key == ord('l'):
            class_name = "Slipped"

cap.release()
cv2.destroyAllWindows()
