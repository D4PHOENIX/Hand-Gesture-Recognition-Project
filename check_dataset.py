import pickle
import os
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

img_dir = 'D:\\University\\Programming\\Python\\AI Projects\\Sign Language Translator\\Data'
output_dir = 'processed_images'  # Define a folder to save images

# Create the output folder if it doesn't exist
output_dir_path = os.path.join(img_dir, output_dir)
os.makedirs(output_dir_path, exist_ok=True)

# Debug: Check the root directory contents
print(f"Contents of the dataset directory: {os.listdir(img_dir)}")

data = []
labels = []

for dir_ in os.listdir(img_dir):
    dir_path = os.path.join(img_dir, dir_)
    # Debug: Check each item in the root directory
    print(f"Checking {dir_}: Is directory? {os.path.isdir(dir_path)}")

    if os.path.isdir(dir_path):
        print(f"Contents of the directory {dir_}: {os.listdir(dir_path)}")
        for img_path in os.listdir(dir_path):
            data_aux = []

            x_ = []
            y_ = []

            load_img = cv2.imread(os.path.join(dir_path, img_path))

            if load_img is None:
                print(f"Failed to load image: {img_path}")
            else:
                print(f"Loaded image: {img_path}")
                rgb_img = cv2.cvtColor(load_img, cv2.COLOR_BGR2RGB)
                dataset_img = hands.process(rgb_img)
                if dataset_img.multi_hand_landmarks:
                    for hand_landmarks in dataset_img.multi_hand_landmarks:
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

                    data.append(data_aux)
                    labels.append(dir_)

                    # Draw hand landmarks on the image
                    for hand_landmarks in dataset_img.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            rgb_img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Save processed image with hand landmarks drawn
                    output_img_path = os.path.join(output_dir_path, f"{dir_}_{img_path}")
                    cv2.imwrite(output_img_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                    print(f"Saved processed image: {output_img_path}")

data_dict = {'data': data, 'labels': labels}
print(data_dict)
print("Data saved successfully !")
