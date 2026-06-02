import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
import torch.nn.functional as F

class AiDetectorCNN(nn.Module):
    def __init__(self):
        super(AiDetectorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
def extract_and_save_roi(video_path, output_folder):
    print(f"--- Starting ROI Extraction ---")
    
    # 1. Check if output directory exists
    if not os.path.exists(output_folder):
        print(f"Creating output directory: {output_folder}")
        os.makedirs(output_folder)
    else:
        print(f"Output directory exists: {output_folder}")

    # 2. Configure Face Detector
    print(f"Initializing Face Detector with model: blaze_face_short_range.tflite")
    base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    
    # 3. Load Video
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file.")
        return

    frame_count = 0
    roi_count = 0

    with vision.FaceDetector.create_from_options(options) as detector:
        print("Detector initialized successfully. Starting frame processing...")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print(f"Reached end of video or failed to read frame {frame_count}.")
                break

            # 4. Process Frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_result = detector.detect(mp_image)

            # 5. Extract and Save
            if detection_result.detections:
                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    face_roi = frame[bbox.origin_y:bbox.origin_y+bbox.height, 
                                     bbox.origin_x:bbox.origin_x+bbox.width]
                    
                    if face_roi.size > 0:
                        save_path = f"{output_folder}/face_{roi_count:04d}.jpg"
                        cv2.imwrite(save_path, face_roi)
                        roi_count += 1
                        if roi_count % 10 == 0: # Print status every 10 faces
                            print(f"Extracted {roi_count} faces so far...")
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
    
    print(f"--- Finished! Processed {frame_count} frames. Extracted {roi_count} face ROIs. ---")

if __name__ == "__main__":
    extract_and_save_roi('data/raw/1000547533.mp4', 'data/processed/faces')