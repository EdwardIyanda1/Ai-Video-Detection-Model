import cv2
import os

def extract_frames(video_path, output_folder):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame as a JPEG image
        frame_name = f"{output_folder}/frame_{frame_count:04d}.jpg"
        cv2.imwrite(frame_name, frame)
        frame_count += 1

    cap.release()
    print(f"Successfully extracted {frame_count} frames to '{output_folder}'.")

# Example usage (ensure your video file is in the same directory)
# extract_frames('1000547533.mp4', 'extracted_frames')