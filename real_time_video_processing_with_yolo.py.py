import cv2
from ultralytics import YOLO
import numpy as np
import os
from pathlib import Path
import time

# Load YOLO model
model = YOLO('MODEL_PATH/best_model.pt')

# Video folder
video_folder = 'VIDEO_FOLDER_PATH'
output_folder = 'OUTPUT_FOLDER_PATH'

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Supported video formats
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

def draw_boxes(frame, results):
    # For each detection
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 coordinates
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # For each box
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if conf > 0.25:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box)
                
                # Class label - abbreviated format
                label = f"{model.names[cls_id][:10]}-{conf:.2f}"
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Smaller font and background for label
                font_scale = 0.5
                thickness = 1
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                # Label position adjustments
                padding = 3
                
                if x1 + text_size[0] > frame.shape[1]:
                    label_x = frame.shape[1] - text_size[0]
                else:
                    label_x = x1
                
                if y1 < text_size[1] + 5:
                    label_y = y2 + text_size[1] + padding
                    rect_y1 = y2 + padding
                    rect_y2 = y2 + text_size[1] + (padding * 2)
                else:
                    label_y = y1 - padding
                    rect_y1 = y1 - text_size[1] - (padding * 2)
                    rect_y2 = y1 - padding
                
                cv2.rectangle(frame, 
                            (label_x, rect_y1), 
                            (label_x + text_size[0], rect_y2), 
                            (0, 255, 0), 
                            -1)
                
                cv2.putText(frame, 
                          label, 
                          (label_x, label_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          font_scale, 
                          (0, 0, 0), 
                          thickness)
    return frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}!")
        return

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Parameters for resizing
    max_width, max_height = 640, 480
    scale = min(max_width / frame_width, max_height / frame_height)
    new_width = int(frame_width * scale)
    new_height = int(frame_height * scale)

    # Output video file settings
    video_name = Path(video_path).stem
    output_path = os.path.join(output_folder, f'{video_name}_tested.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (new_width, new_height))

    # Use a single window
    window_name = "Video Processing"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, new_width, new_height)
    cv2.moveWindow(window_name, 0, 0)

    print(f"\nProcessing: {video_name}")
    print(f"Total frames: {total_frames}")

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Show progress every 30 frames
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - FPS: {fps:.1f}", end='\r')
        
        frame = cv2.resize(frame, (new_width, new_height))
        results = model(frame)
        annotated_frame = draw_boxes(frame, results)
        
        # Video name and progress on frame
        text = f"{video_name} - %{(frame_count/total_frames)*100:.1f}"
        cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        out.write(annotated_frame)
        cv2.imshow(window_name, annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nProcess stopped by user.")
            cap.release()
            out.release()
            return False

    cap.release()
    out.release()
    print(f"\n{video_name} processed!")
    return True

def main():
    video_files = [f for f in os.listdir(video_folder) 
                   if os.path.isfile(os.path.join(video_folder, f)) 
                   and f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"Warning: No videos found in {video_folder}!")
        return

    print(f"Found {len(video_files)} videos.")
    
    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(video_folder, video_file)
        print(f"\n[{i}/{len(video_files)}] Processing video: {video_file}")
        
        if not process_video(video_path):
            break
        
        time.sleep(1)  # Short wait between videos

    cv2.destroyAllWindows()
    print("\nAll videos processed!")

if __name__ == "__main__":
    main()
