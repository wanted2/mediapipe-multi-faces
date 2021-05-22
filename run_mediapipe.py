import cv2
import mediapipe as mp
from time import time
file_list = ['selfie.jpg', 'selfie-3.jpg', 'selfie-small.jpg']
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For static images:
with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:
  for idx, file in enumerate(file_list):
    print(f'Processing {file} ...')
    image = cv2.imread(file)
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    t1 = time()
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    t2 = time()
    print(f'Processed {file} in {t2-t1:.5f} second(s)')

    # Draw face detections of each face.
    if not results.detections:
      continue
    annotated_image = image.copy()
    for detection in results.detections:
      print('Nose tip:')
      print(mp_face_detection.get_key_point(
          detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
      mp_drawing.draw_detection(annotated_image, detection)
    cv2.imwrite('./annotated_image' + str(idx) + '.png', annotated_image)
