import cv2

def list_video_capture_devices():
    # Try using different backends to list available devices
    index = 0
    arr = []
    
    # Try using CAP_DSHOW backend
    print("Checking devices using CAP_DSHOW backend...")
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    
    if len(arr) > 0:
        print("Available video capture devices using CAP_DSHOW:", arr)
        return

    # Try using CAP_MSMF backend
    index = 0
    arr = []
    print("Checking devices using CAP_MSMF backend...")
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1

    if len(arr) > 0:
        print("Available video capture devices using CAP_MSMF:", arr)
        return

    # Try using CAP_V4L2 backend (Linux)
    index = 0
    arr = []
    print("Checking devices using CAP_V4L2 backend...")
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1

    if len(arr) > 0:
        print("Available video capture devices using CAP_V4L2:", arr)
        return

    print("No available video capture devices found.")

list_video_capture_devices()
