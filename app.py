import streamlit as st
import cv2
import requests
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.description = ""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.cap = cv2.VideoCapture(0)
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.process_every_n_frames = max(1, fps // 2)

    def get_description_from_api(self, frame):
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post("https://9e2c-34-41-55-211.ngrok-free.app/process_frame/", files={"file": img_encoded.tobytes()})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None

    def wrap_text(self, text, font, max_width):
        if text is None:
            return ["Error in generating description"]
        lines = []
        for line in text.split('\n'):
            words = line.split(' ')
            current_line = words[0]
            for word in words[1:]:
                if cv2.getTextSize(current_line + ' ' + word, font, 0.7, 2)[0][0] < max_width:
                    current_line += ' ' + word
                else:
                    lines.append(current_line)
                    current_line = word
            lines.append(current_line)
        return lines

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.frame_count % self.process_every_n_frames == 0:
            result = self.get_description_from_api(img)
            if result and 'error' not in result:
                self.description = result.get("description", "")
                self.fall_probability = result.get("fall_probability", 0)
                self.emotion_description = result.get("emotion", "")
                self.action = result.get("action", "")
                self.region = result.get("region", None)
                self.objects = result.get("objects", [])
                self.pose_landmarks = result.get("pose_landmarks", [])
                self.alert_message = result.get("alert_message", "")

        full_description = f"{self.description}\nFall probability: {self.fall_probability}%\nEmotion: {self.emotion_description}\nAction: {self.action}\nObjects: {', '.join(obj['class'] for obj in self.objects)}"
        if self.alert_message:
            full_description += f"\nAlert: {self.alert_message}"

        for obj in self.objects:
            class_name = obj['class']
            cv2.putText(img, class_name, (obj['xmin'], obj['ymin'] - 10), self.font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(img, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (0, 255, 0), 2)

        if self.pose_landmarks:
            pose_landmark_list = []
            for lm in self.pose_landmarks:
                landmark = landmark_pb2.NormalizedLandmark(
                    x=float(lm['x']),
                    y=float(lm['y']),
                    z=float(lm['z'])
                )
                pose_landmark_list.append(landmark)

            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList(landmark=pose_landmark_list)

            mp_drawing.draw_landmarks(
                img,
                pose_landmarks_proto,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

        if self.region:
            x, y, w, h = self.region['x'], self.region['y'], self.region['w'], self.region['h']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, self.emotion_description, (x, y - 10), self.font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        white_canvas = 255 * np.ones(shape=[img.shape[0], int(img.shape[1] * 0.3), 3], dtype=np.uint8)
        max_text_width = int(img.shape[1] * 0.3) - 20
        wrapped_text = self.wrap_text(full_description, self.font, max_text_width)

        y0, dy = 50, 30
        for i, line in enumerate(wrapped_text):
            y = y0 + i * dy
            if "Alert" in line:
                cv2.putText(white_canvas, line, (10, y), self.font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(white_canvas, line, (10, y), self.font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        combined_frame = np.hstack((img, white_canvas))

        self.frame_count += 1
        return combined_frame

st.title("AuPair Vision Model")

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
