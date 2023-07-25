import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle

with open('body_language1.pkl', 'rb') as f:
    model = pickle.load(f)

# Load CSV file containing body activity labels for each frame
activity_data = pd.read_csv('coords1.csv')

# Function to get the activity label based on frame number
def get_activity_label(frame_num):
    return activity_data.loc[activity_data['Frame'] == frame_num, 'Activity'].values[0]

video_path = 't 1.mp4'  # Replace 'your_video_file.mp4' with your video file path
cap = cv2.VideoCapture('videos3/smash-and-grab robbery.mp4')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Video ended.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            X = pd.DataFrame([pose_row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class, body_language_prob)

            coords = tuple(np.multiply(
                np.array((
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                [1080, 720]).astype(int))

            cv2.rectangle(image,
                          (coords[0], coords[1] + 5),
                          (coords[0] + len(body_language_class) * 20, coords[1] - 30),
                          (245, 117, 16), -1)
            cv2.putText(image, body_language_class, coords,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

            cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0],
                        (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Get the frame number from the video capture
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Get the activity label for the current frame from the CSV file
            activity_label = get_activity_label(frame_num)

            # Calculate the position to display the activity label at the top-right corner
            activity_coords = (image.shape[1] - 250, 40)

            # Draw a background rectangle for the label
            cv2.rectangle(image, (activity_coords[0], activity_coords[1] - 30),
                          (activity_coords[0] + len(activity_label) * 10, activity_coords[1] + 10),
                          (245, 117, 16), -1)

            # Draw the activity label at the top-right corner of the frame
            cv2.putText(image, 'Activity:', (activity_coords[0], activity_coords[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, activity_label, (activity_coords[0], activity_coords[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        except:
            pass

        cv2.imshow('Raw Video Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
