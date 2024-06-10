---
layout: post
title:  "Real-time hand tracking with Mediapipe and CV2"
date:   2024-06-09 12:54:15 +0800
categories: jekyll update
---

In this project, I aim to build a handtracking app. The reason for doing these projects is to explore computer vision libraries like Mediapipe which provides real-time hand tracking capabilities and gestures and CV2 which will help in drawing styles on the image. 

They have trained the model on different hand gestures and it can pick up different points on the fingers. I want to add a draw feature which lets you draw real time in the video, seems like it would be a good use case for video conferencing apps like zoom or google meet. 


## Approach
### Handtracking app
The app will be powered by Mediapipe which is an open source library developed by google. I will use the library to track and draw in a video. First step will be to build the hand recognition then the drawing.  

```
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
```

The drawing utils will let us draw lines in a live video, drawing styles will help customize with different colors and hands will recognize the hand gestures by each finger. 
On the backend, mediapipe is using cv2 to draw lines on pixel coordinates. It uses 3 main concepts to do this:
1. Proto Messages: It uses protocol buffers (protobuf) to serialize structured data. 
2. Normalized Coordinates: Coordinates normalized to the range [0, 1], allowing image-independent representation of positions.
3. Drawing Specifications: Allows customization of drawing attributes, making the functions flexible for various use cases.

The hands class uses Handlandmarks which is 21 points on the hand marked with indices. This is useful for referring to specific landmarks by name rather than by number. It uses calculators which handles various tasks such as converting images to tensors, running inference, and extracting landmarks.

#### Function to get index to coordinate mapping
This function will take the image and the hand detection result, then maps each landmark index from hand detection to corresponding pixel in the image in a dictionary. 
```
def get_idx_to_coordinates(image, results):
    idx_to_coordinates = {} 
    height, width, _ = image.shape 
    for hand_landmarks in results.multi_hand_landmarks:
        for idx, landmark in enumerate(hand_landmarks.landmark):
            x, y = int(landmark.x * width), int(landmark.y * height)
            idx_to_coordinates[idx] = (x, y)
    return idx_to_coordinates
```

We write an additional function which will recognize the fist gesture to stop the drawing. We will use this in the Handtracking class defined later. 
```
def is_fist(hand_landmarks):
    fist_threshold = 0.2
    for landmark in hand_landmarks.landmark:
        if landmark.y < hand_landmarks.landmark[0].y - fist_threshold:
            return False
    return True
```

#### Handtracking class
This class will be used to track hand movements in video frames, draw landmarks and draw lines with the index finger. 
To store the tracked points from the hand, we need to initialize a deque for addition and removal of points. This will be to store and keep the drawing in the video frame. 

```
class HandTracking:
    def __init__(self):
        self.pts = deque(maxlen=512)
        self.draw = True
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
```
```
    def process_frame(self, frame):
        idx_to_coordinates = {}
        
        image = cv2.flip(frame, 1)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape
            
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if is_fist(hand_landmarks):
                    self.draw = False
                    self.pts.clear()
                else:
                    self.draw = True
                
                if self.draw:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    idx_to_coordinates = get_idx_to_coordinates(image, results)

        if self.draw and 8 in idx_to_coordinates:
            self.pts.appendleft(idx_to_coordinates[8])  # Index Finger
            
        smoothed_pts = smooth_points(list(self.pts))

        for i in range(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue
            thick = int(np.sqrt(len(self.pts) / float(i + 1)) * 2)
            cv2.line(image, self.pts[i - 1], self.pts[i], (0, 0, 255), thick)  # Change to red

        return image

```
The process frame function will process a single video frame to detect hands and draw landmarks and tracking lines.
First step is preprocessing, that is to flip the image horizontally to match the typical front facing camera view. Then to convert the image color from BGR to RGD. This can be done with openCV. Then we can process the image using the mediapipe hands process method. 

Then we iterate over each detected hand landmark and draw on the image using mp_drawing.draw_landmarks and update the landmark positions using the get_idx_to_coordinates function defined earlier.
Then if the index finger is present in the idx_to_coordinate dictionary, we add the points to deque. 

Finally, we can iterate over the deque to draw the line with cv2.line and return the image. 

## Result
To wrap all this in a flask application and run it: 
```
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from flask import Flask, Response, render_template, render_template_string
import threading
from flask_cors import CORS

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

app = Flask(__name__)
CORS(app)

# Function to get index to coordinates mapping
def get_idx_to_coordinates(image, results):
    idx_to_coordinates = {}
    height, width, _ = image.shape
    for hand_landmarks in results.multi_hand_landmarks:
        for idx, landmark in enumerate(hand_landmarks.landmark):
            x, y = int(landmark.x * width), int(landmark.y * height)
            idx_to_coordinates[idx] = (x, y)
    return idx_to_coordinates

# Function to check if the hand is in a fist position
def is_fist(hand_landmarks):
    fist_threshold = 0.2
    for landmark in hand_landmarks.landmark:
        if landmark.y < hand_landmarks.landmark[0].y - fist_threshold:
            return False
    return True


class HandTracking:
    def __init__(self):
        self.pts = deque(maxlen=512)  # Increase the maxlen to store more points
        self.draw = True
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
    
    def process_frame(self, frame):
        idx_to_coordinates = {}
        
        image = cv2.flip(frame, 1)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape
            
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if is_fist(hand_landmarks):
                    self.draw = False
                    self.pts.clear()
                else:
                    self.draw = True
                
                if self.draw:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    idx_to_coordinates = get_idx_to_coordinates(image, results)

        if self.draw and 8 in idx_to_coordinates:
            self.pts.appendleft(idx_to_coordinates[8])  # Index Finger

        for i in range(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue
            thick = int(np.sqrt(len(self.pts) / float(i + 1)) * 2)
            cv2.line(image, self.pts[i - 1], self.pts[i], (0, 0, 255), thick)  # Change to red

        return image

hand_tracker = HandTracking()

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = hand_tracker.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

```

[Clone this project](https://github.com/Anushk97/handtracking2)

***Thanks for reading!***