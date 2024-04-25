import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time

# Load the MoveNet model from TensorFlow Hub.
model = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-lightning/4")
movenet = model.signatures['serving_default']

def draw_all_keypoints(frame, keypoints):
    for keypoint in keypoints[0, 0]:
        x, y = int(keypoint[1] * frame.shape[1]), int(keypoint[0] * frame.shape[0])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw green circles on keypoints

    return frame

def process_frame(frame):
    resized_frame = cv2.resize(frame, (192, 192))
    resized_frame = np.expand_dims(resized_frame, axis=0)
    resized_frame = tf.cast(resized_frame, dtype=tf.int32)

    # Get the keypoints from the MoveNet model.
    outputs = movenet(resized_frame)
    keypoints = outputs['output_0']

    frame_with_keypoints = draw_all_keypoints(frame.copy(), keypoints)
    return frame_with_keypoints

# For real time pass 0 or 1 according to camera and for video pass video address.
cap = cv2.VideoCapture("girl_dance.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    processed_frame = process_frame(frame)
    print("Frame resolution:", frame.shape[1], "x", frame.shape[0])

    cv2.imshow('Frame with Keypoints', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
