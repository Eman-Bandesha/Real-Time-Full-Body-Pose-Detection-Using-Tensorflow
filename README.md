# Real Time Full Body Pose Detection Using Tensorflow Model (Lightweight)

This Python script performs full-body pose detection using the MoveNet model, a lightweight TensorFlow model. It is designed for single-person pose estimation and provides fast and accurate results. The script processes each frame of a video to detect keypoints representing the full-body pose. It takes around 0.08 seconds to detect keypoints on an image of high resolution (360x640) on a CPU, making it suitable for real-time applications.
The MoveNet model is efficient and provides accurate pose estimation, making it a preferred choice for applications requiring fast and reliable full-body pose detection. You can find the MoveNet model on TensorFlow Hub [here]([https://tfhub.dev/google/movenet/singlepose/lightning/4](https://www.kaggle.com/models/google/movenet/tensorFlow2)).

## Dependencies

- Python 3.x
- OpenCV (cv2)
- TensorFlow
- TensorFlow Hub
- NumPy


## Functionality

The script performs the following tasks:

1. Loads the MoveNet model from TensorFlow Hub.
2. Reads each frame from the input video file.
3. Processes each frame to detect keypoints representing the full-body pose.
4. Draws the detected keypoints on the frame.
5. Writes the processed frames with keypoints drawn to an output video file.

## Results
The resultant video is [here](https://github.com/Eman-Bandesha/Real-Time-Full-Body-Pose-Detection-Using-Tensorflow/blob/main/output_video.mp4).

## Acknowledgments

This script utilizes the MoveNet model from TensorFlow Hub for full-body pose detection. The MoveNet model is available on TensorFlow Hub at [this link](https://www.kaggle.com/models/google/movenet/tensorFlow2). We would like to express our gratitude to the TensorFlow team for providing this pre-trained model.

