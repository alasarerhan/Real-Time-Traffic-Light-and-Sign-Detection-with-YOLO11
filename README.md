# 🚦 YOLO Road Sign Detection

A real-time road sign detection system using YOLO  that processes video files and annotates road signs with bounding boxes. The system includes features like progress tracking, FPS monitoring, and adaptive label positioning.

## 🎯 Features

- Real-time road sign detection with confidence thresholding
- Support for multiple video formats (.mp4, .avi, .mov, .mkv)
- Automatic video resizing for optimal performance
- Batch processing of multiple videos
- Live preview during processing

## 🛠️ Requirements

### Dependencies

```bash
ultralytics  # For YOLO implementation
opencv-python  # For video processing
numpy         # For numerical operations
```

Install dependencies using:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
└── root/
    ├── real_time_video_processing_with_yolo.py                 # Main processing script
    ├── best_model.pt                                           # Trained YOLO model
    └── traffic-lights-and-signs-recognition-with-yolo.ipynb    # Model training 
```

## ⚙️ Configuration

Before running the script, configure these paths in `real_time_video_processing_with_yolo.py`:

```python
# Load YOLO model
model = YOLO('MODEL_PATH/best_model.pt')

# Video folder paths
video_folder = 'VIDEO_FOLDER_PATH'
output_folder = 'OUTPUT_FOLDER_PATH'
```

## 🚀 Usage

1. Place your input videos in the configured video folder
2. Run the script:
```bash
python real_time_video_processing_with_yolo.py
```

### Processing Features

- **Confidence Threshold**: Set to 0.35 by default
- **Output Resolution**: Automatically scaled to max 640x480
- **Progress Monitoring**: Updates every 30 frames
- **Video Preview**: Real-time display of processed frames
- **Batch Processing**: Automatically processes all videos in the input folder

### Controls

- Press `q` to stop processing and exit
- Close the preview window to proceed to the next video

## 🎥 Output

The script generates processed videos with:
- Bounding boxes around detected road signs
- Confidence scores for each detection
- Progress indicator
- Video name overlay

Output files are saved as `{original_name}_tested.mp4` in the specified output folder.

## 💻 Implementation Details

### Key Functions

#### `draw_boxes(frame, results)`
- Draws bounding boxes and labels for detected objects
- Implements smart label positioning to avoid screen edges
- Uses color-coded boxes with confidence scores

#### `process_video(video_path)`
- Handles individual video processing
- Implements progress tracking and FPS calculation
- Manages video resizing and output generation

#### `main()`
- Orchestrates batch processing of videos
- Handles file discovery and process flow
- Implements error handling and user interruption

### Performance Optimizations

- Automatic video resizing for consistent performance
- Batch frame processing
- Efficient memory management
- Progress tracking with minimal overhead

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO implementation
- OpenCV community for computer vision tools
