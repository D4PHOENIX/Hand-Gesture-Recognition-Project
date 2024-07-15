import pickle
import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time

# Load the trained model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Try different camera indices
def open_camera(indices):
    for index in indices:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap, index
    return None, -1

# List of indices to try
camera_indices = [3]  # Update the list with other camera indices if needed
cap, used_index = open_camera(camera_indices)

if not cap or used_index == -1:
    print("Error: Could not open any video device")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.5)

# Update the labels_dict to match your actual labels
labels_dict = {0: 'play/pause', 1: 'next song', 2: 'previous song'}
inverse_labels_dict = {v: k for k, v in labels_dict.items()}  # To get labels from predictions

# Define the functions for each action
def play_pause():
    print("Playing/Pausing the song")
    # Simulate media play/pause key press
    keyboard.send('play/pause media')

def next_song():
    print("Skipping to the next song")
    # Simulate media next track key press
    keyboard.send('next track')

def previous_song():
    print("Going to the previous song")
    # Simulate media previous track key press
    keyboard.send('previous track')

# Function to execute the action based on the detected gesture
def perform_action(gesture):
    if gesture == "play/pause":
        play_pause()
    elif gesture == "next song":
        next_song()
    elif gesture == "previous song":
        previous_song()
    else:
        print("Unknown Gesture")

# Timing variables
last_detected_time = time.time()
detection_interval = 1  # Time in seconds

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Continuous visualization of hand landmarks and text box
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Ensure the length of data_aux is 42 to match the training data
        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('Media Control', frame)

    # Perform gesture detection and action execution every 5 seconds
    current_time = time.time()
    if current_time - last_detected_time >= detection_interval:
        last_detected_time = current_time

        if results.multi_hand_landmarks:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            print(f"Detected Gesture: {predicted_character}")
            perform_action(predicted_character)

    if cv2.waitKey(1) & 0xFF == 32:  # Exit on spacebar press
        break

cap.release()
cv2.destroyAllWindows()
