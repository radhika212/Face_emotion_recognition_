import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

class EmotionRecognition:
    def __init__(self):
        # Create emotion queue of last 'x' emotions to smooth the output
        self.emotion_queue = deque(maxlen=10)
        self.model = load_model('model.h5')  # Load your new .h5 model file here

    def smooth_emotions(self, prediction):
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        emotion_values = {emotion: 0.0 for emotion in emotions}

        emotion_probability, emotion_index = max((val, idx) for (idx, val) in enumerate(prediction[0]))
        emotion = emotions[emotion_index]

        # Increase the weight for "Surprise" to reduce likelihood of it being categorized as "Neutral"
        if emotion == "Surprise":
            emotion_probability *= 1.2  # Boost "Surprise" predictions slightly

        # Append the new emotion and if the max length is reached pop the oldest value out
        self.emotion_queue.appendleft((emotion_probability, emotion))

        # Iterate through each emotion in the queue and create an average of the emotions
        for pair in self.emotion_queue:
            emotion_values[pair[1]] += pair[0]

        # Select the current emotion based on the one that has the highest value
        average_emotion = max(emotion_values.items(), key=lambda item: item[1])[0]
        average_probability = emotion_values[average_emotion] / len(self.emotion_queue)

        return average_emotion, average_probability

    def preprocess_image(self, roi_gray):
        # Normalize the image to match the training preprocessing
        image_scaled = np.array(cv2.resize(roi_gray, (48, 48)), dtype=float) / 255.0
        image_processed = image_scaled.reshape([-1, 48, 48, 1])
        return image_processed

    def process_image(self, roi_gray, img):
        image_processed = self.preprocess_image(roi_gray)
        prediction = self.model.predict(image_processed)
        emotion, probability = self.smooth_emotions(prediction)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Emotion: {emotion} ({probability * 100:.2f}%)", (50, 450), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('img', img)

    def run(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)

        while True:
            ret, img = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                self.process_image(roi_gray, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    emotion_recognition = EmotionRecognition()
    emotion_recognition.run()
