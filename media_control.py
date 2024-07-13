import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Load the trained model
model = pickle.load(open('trained_model.pickle', 'rb'))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Process hand landmarks
            landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark]).flatten()
            prediction = model.predict([landmarks])

            # Perform actions based on predictions
            gesture = prediction[0]

            if gesture == 'play':
                pyautogui.press('playpause')  # Example: Play or pause media
            elif gesture == 'pause':
                pyautogui.press('playpause')  # Example: Play or pause media
            elif gesture == 'skip':
                pyautogui.hotkey('ctrl', 'right')  # Example: Skip forward in media

    cv2.imshow('Gesture-Based Media Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
