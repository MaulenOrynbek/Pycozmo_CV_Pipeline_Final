Cozmo Emotion Mimicry (Closest Face Tracking)
FILE: cozmo_pipeline_final.py
To use this code just download the repository and run the code. Better to run the code on a device with a better processor. 
For face detection YOLOv8 is used. For facial emotion recognition, this is a model based on the ConvNeXt V2 Large architecture. The DrGM/DrGM-ConvNeXt-V2L-Facial-Emotion-Recognition model was fine-tuned on a custom "Facial Emotion Expressions dataset" with 62,923 images across the 7 basic emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise).
This code enables a Cozmo Robot to act as an emotional mirror. Using its built-in camera, Cozmo monitors everyone in the room, identifies their emotions using a fine-tuned model, but physically mimics only the person closest to him. 
To ensure smooth interaction, Cozmo includes a 2-second "thought" buffer between expression changes.


🚀 Features
Multi-Face Detection: Detects and labels every face in the frame.

Emotion Recognition: Classifies faces into 7 emotions: Angry, Disgusted, Fear, Happy, Sad, Surprise, and Neutral.

Proximity Logic: Automatically selects the "Target" based on the largest bounding box (the person physically closest to the camera).

Rate Limiting: A 2-second cooldown prevents Cozmo from flickering between expressions too rapidly.

Live Debug UI: An OpenCV window showing green boxes for the target and blue boxes for others.


⚙️ How It Works
1. Proximity Detection
The script calculates the area of each detected face: Area = Width * Height. The face with the largest area is assigned as the closest_face_idx.
2. Emotion Processing
While all faces are analyzed and logged via JSON for data collection, only the emotion data from the "closest" index is passed to the robot's logic.
3 Cooldown Mechanism
To prevent erratic behavior, the script uses time.time(). Cozmo will only update his face if:The detected emotion of the target has changed.At least 2.0 seconds have passed since the last update.



Metric	                   Value
Accuracy	               90.43%
F1 Score (Weighted)	       0.9042
Validation Loss	           0.7031
Inference Time (Batch)	   23.63s (Total)
Throughput	               532.68 samples/sec

Confusion Matrix is in the Confusion_Matrix.png
