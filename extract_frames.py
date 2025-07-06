import os
import cv2

# Paths
dataset_path = "FakeAVCeleb"  # Change this if needed
output_path = "dataset/extracted_frames"

# Function to extract frames
def extract_frames(source_folder, target_folder, num_frames=10):
    """
    Extracts frames from videos and saves them as images.
    :param source_folder: Path to folder containing video files
    :param target_folder: Path to save extracted images
    :param num_frames: Number of frames to extract per video
    """
    for subfolder in os.listdir(source_folder):  # Loop through subfolders
        subfolder_path = os.path.join(source_folder, subfolder)
        if os.path.isdir(subfolder_path):  # Ensure it's a folder
            for video_name in os.listdir(subfolder_path):  # Loop through videos
                video_path = os.path.join(subfolder_path, video_name)
                
                # Open video
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                success = True
                
                while success and frame_count < num_frames:
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    frame_filename = f"{video_name}_frame{frame_count}.jpg"
                    cv2.imwrite(os.path.join(target_folder, frame_filename), frame)
                    frame_count += 1
                
                cap.release()
                print(f"Extracted {frame_count} frames from {video_name}")

# Extract frames from real and fake videos
extract_frames(f"{dataset_path}/real", f"{output_path}/real")
extract_frames(f"{dataset_path}/fake", f"{output_path}/fake")

print("✅ Frames extracted successfully!")
