import face_recognition
import os, sys
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def face_Confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else: 
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    def __init__(self):
        self.process_current_frame = True  
        self.encode_faces()
        self.confidence_data = []  # List to store confidence levels
    
    def encode_faces(self):
        self.known_face_encodings = []  
        self.known_face_names = []  

        for image in os.listdir('Faces'):
            if image.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                face_image = face_recognition.load_image_file(f'Faces/{image}')
                face_encodings = face_recognition.face_encodings(face_image)

                if len(face_encodings) > 0:
                    face_encoding = face_encodings[0]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(image)
                    print(f"Added {image} to the dataset.")

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit("video source not found") 

        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distance = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distance) 

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_Confidence(face_distance[best_match_index])
                        self.confidence_data.append(round(float(confidence.strip('%')), 2))  # Collect confidence data

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)

                text = f'{name}' 
                cv2.putText(frame, text, (left + 6, bottom - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('face recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def plot_confidence_graph(self):
        plt.figure(figsize=(8, 6))
        plt.hist(self.confidence_data, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Confidence Level')
        plt.ylabel('Frequency')
        plt.title('Distribution of Confidence Levels')
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
    fr.plot_confidence_graph()
