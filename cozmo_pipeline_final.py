import cv2
import numpy as np
import pycozmo
from PIL import Image as PILImage
import time
import json
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO

# ====================== LOCAL OFFLINE MODELS ======================
# Change these paths if your folders/files are named differently
EMOTION_MODEL_DIR = "models/emotion_convnext"          # Folder from snapshot_download
YOLO_FACE_MODEL_PATH = "models/yolo_face/model.pt"     # Path to downloaded model.pt

print("Loading local emotion model...")
processor = AutoImageProcessor.from_pretrained(EMOTION_MODEL_DIR, local_files_only=True)
emotion_model = AutoModelForImageClassification.from_pretrained(EMOTION_MODEL_DIR, local_files_only=True)

print("Loading local YOLO face detection model...")
face_model_yolo = YOLO(YOLO_FACE_MODEL_PATH)

print("Models loaded successfully!")

# ==============================================================================

EMOTION_TO_EXPRESSION = {
    "happy": pycozmo.expressions.Happiness(),
    "sad": pycozmo.expressions.Sadness(),
    "angry": pycozmo.expressions.Anger(),
    "surprise": pycozmo.expressions.Surprise(),
    "disgust": pycozmo.expressions.Disgust(),
    "fear": pycozmo.expressions.Fear(),
    "neutral": pycozmo.expressions.Neutral(),
}

last_im = None
frame_id = 0
last_emotion_time = 0
COOLDOWN_SECONDS = 2.0
current_robot_emotion = None

def on_camera_image(cli, new_im):
    global last_im
    last_im = np.array(new_im)

with pycozmo.connect(enable_procedural_face=False) as cli:
    angle = (pycozmo.robot.MAX_HEAD_ANGLE.radians - pycozmo.robot.MIN_HEAD_ANGLE.radians) / 2.0
    cli.set_head_angle(angle)
    cli.enable_camera(color=True)
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)

    cv2.namedWindow("Cozmo Camera (YOLO Face + ConvNeXt Emotion - Offline)", cv2.WINDOW_NORMAL)

    timer = pycozmo.util.FPSTimer(10)

    while True:
        if last_im is not None:
            frame_id += 1
            frame = last_im.copy()  # RGB, 320x240 from Cozmo

            # === IMPROVED YOLO DETECTION FOR SMALL COZMO FRAMES ===
            # Upscale 2x to help YOLO detect small faces
            frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

            # Run YOLO with lower confidence for small faces
            results = face_model_yolo(frame_resized, conf=0.25, imgsz=640, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] in resized coords

            # Scale bounding boxes back to original 320x240 resolution
            if len(boxes) > 0:
                boxes[:, [0, 2]] *= 320 / 640  # x coordinates
                boxes[:, [1, 3]] *= 240 / 480  # y coordinates
                boxes = boxes.astype(int)

            max_area = 0
            closest_face_idx = -1
            detected_emotions = []

            for i, (x1, y1, x2, y2) in enumerate(boxes):
                w = x2 - x1
                h = y2 - y1
                area = w * h

                if area > max_area:
                    max_area = area
                    closest_face_idx = i

                # Crop face from original frame (better color/quality)
                face_roi = frame[max(0, y1):min(240, y2), max(0, x1):min(320, x2)]

                if face_roi.size == 0:
                    continue  # Skip invalid crops

                # Emotion prediction
                face_pil = PILImage.fromarray(face_roi)
                inputs = processor(images=face_pil, return_tensors="pt")
                with torch.no_grad():
                    outputs = emotion_model(**inputs)
                    pred_idx = outputs.logits.argmax(-1).item()
                    emotion = emotion_model.config.id2label[pred_idx]  # e.g., "Happy"
                    confidence = torch.softmax(outputs.logits, dim=-1)[0][pred_idx].item()

                detected_emotions.append(emotion)

                # UI Overlay
                is_closest = (i == closest_face_idx)
                color = (0, 255, 0) if is_closest else (255, 0, 0)  # Green = target
                thickness = 4 if is_closest else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                label = f'{emotion} ({confidence:.2f})'
                if is_closest:
                    label += " (Target)"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)

                print(json.dumps({
                    "frame": frame_id,
                    "id": i,
                    "emotion": emotion,
                    "confidence": round(confidence, 2),
                    "closest": is_closest
                }))

            # === Cozmo Update (only closest face) ===
            current_time = time.time()
            if closest_face_idx != -1:
                target_emotion = detected_emotions[closest_face_idx]
                emotion_key = target_emotion.lower()

                if emotion_key != current_robot_emotion:
                    if (current_time - last_emotion_time) > COOLDOWN_SECONDS:
                        expression = EMOTION_TO_EXPRESSION.get(emotion_key, pycozmo.expressions.Neutral())
                        face_render = expression.render()
                        np_face = np.array(face_render)[::2]
                        cli.display_image(PILImage.fromarray(np_face))

                        current_robot_emotion = emotion_key
                        last_emotion_time = current_time
                        print(f">>> COZMO UPDATED TO: {target_emotion}")

            # Show frame
            cv2.imshow("Cozmo Camera (YOLO Face + ConvNeXt Emotion - Offline)", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        timer.sleep()

    cv2.destroyAllWindows()