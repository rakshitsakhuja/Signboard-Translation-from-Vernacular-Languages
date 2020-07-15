  # Text Detection (Detecting bounding boxes containing text in the images)
   **Data Set:** 967  images with annotations in YOLO format - 2901 images after augmentation <br/>
   **Categories:** 520 English (under represented); 4955 Hindi examples. <br/>            
   **Augmentation:** Flip (Horizontal, Vertical) ; 90° Rotate (Clockwise, Counter-Clockwise, Upside Down) ; Rotation (-15° to +15°) ; Brightness (-30% to +30%) ; Noise (Up to 5%                        of pixels)
  
   **Model:** yolov5s (https://github.com/ultralytics/yolov5) trained on our dataset. <br/>
    Run ***'yolov5s.ipynb'*** for training and inferencing. <br/>
    Use our trained weights - ***last.pt*** for inferencing. Shared here. <br/>
   
