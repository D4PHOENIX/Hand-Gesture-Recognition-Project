import os
import cv2
import mediapipe as mp
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

img_dir = 'D:\\University\\Programming\\Python\\AI Projects\\Sign Language Translator\\Data'

data = []
labels = []

# Process images in batches
batch_size = 100  # Adjust batch size as needed
batch_data = []
batch_labels = []
for dir_ in os.listdir(img_dir):
    for img_path in os.listdir(os.path.join(img_dir, dir_)):
        load_img = cv2.imread(os.path.join(img_dir, dir_, img_path))
        rgb_img = cv2.cvtColor(load_img, cv2.COLOR_BGR2RGB)

        dataset_img = hands.process(rgb_img)
        if dataset_img.multi_hand_landmarks:
            data_aux = []
            for hand_landmarks in dataset_img.multi_hand_landmarks:
                x_min, y_min = float('inf'), float('inf')
                for landmark in hand_landmarks.landmark:
                    x, y = landmark.x, landmark.y
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    data_aux.extend([x - x_min, y - y_min])
            batch_data.append(data_aux)
            batch_labels.append(dir_)

            # Process batch when it reaches batch size
            if len(batch_data) >= batch_size:
                data.extend(batch_data)
                labels.extend(batch_labels)
                batch_data = []
                batch_labels = []

# Process any remaining data
if batch_data:
    data.extend(batch_data)
    labels.extend(batch_labels)

# Save the data and labels
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print('Data saved successfully!')