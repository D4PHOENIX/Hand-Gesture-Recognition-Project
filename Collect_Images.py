import os
import cv2

img_dataset = '.\\Data'
if not os.path.exists(img_dataset):
    os.makedirs(img_dataset)

number_of_classes = 3
dataset_size = 1000

# Function to open the camera
def open_camera(indices):
    for index in indices:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap, index
    return None, -1

# List of indices to try
camera_indices = [3]
cap, used_index = open_camera(camera_indices)

if not cap or used_index == -1:
    print("Error: Could not open any video device")
    exit()

print(f"Using camera index: {used_index}")

for i in range(number_of_classes):
    class_dir = os.path.join(img_dataset, str(i))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(i))

    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        text = 'Ready? Press "Space" ! :)'
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = text_size[1] + 10  # Offset from the top

        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(25) == 32:
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()