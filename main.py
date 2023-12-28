import cv2
import numpy as np
import torch

# Load YOLOv5 model for object detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)  # 'yolov5n' for speed

# Initialize stereo camera
cap = cv2.VideoCapture(0)

# Calculate parameters to reduce the image resolution while maintaining the aspect ratio
desired_width = 640
aspect_ratio = 0.5 * cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
desired_height = int(desired_width / aspect_ratio)

# Init StereoSGBM matcher
window_size = 11
min_disp = 0
num_disp = 48 - min_disp
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    uniquenessRatio=5,
    speckleWindowSize=50,
    speckleRange=16,
    disp12MaxDiff=1,
    P1=8*3*window_size**2,
    P2=32*3*window_size**2
)

object_detection_frequency = 5
frame_counter = 0
last_results_left = None  # Use this to store the last detection results

while True:
    ret, wide_frame = cap.read()
    if not ret:
        break

    # Lower the frame resolution
    resized_frame = cv2.resize(wide_frame, (desired_width * 2, desired_height))

    frame_left = resized_frame[:, :desired_width, :]
    frame_right = resized_frame[:, desired_width:, :]

    if frame_counter % object_detection_frequency == 0:
        results_left = model(frame_left)
        last_results_left = results_left.xyxy[0]  # Store the current detection
    else:
        if last_results_left is not None:
            # Use the last detection results
            for det in last_results_left:
                xmin, ymin, xmax, ymax, conf, cls_id = map(int, det)
                label = model.names[cls_id]
                cv2.rectangle(frame_left, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame_left, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    frame_counter += 1

    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(gray_left, gray_right).astype(np.float32)
    disparity = (disparity - min_disp) / num_disp
    disp_display = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imshow('Object Detection', frame_left)
    cv2.imshow('Disparity Map', disp_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
