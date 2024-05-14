import os
import sys
from dataclasses import dataclass
import cv2
import mediapipe as mp
from src.exception import CustomException
from src.logger import logging
from datetime import datetime

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def overlap(self, box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[0] + box1[2], box2[0] + box2[2])
        y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area

    def merge_boxes(self, boxes, iou_threshold=0.5):
        merged_boxes = []
        for box in boxes:
            if not merged_boxes:
                merged_boxes.append(box)
                continue

            should_merge = False
            for i, merged_box in enumerate(merged_boxes):
                if self.overlap(merged_box, box) > iou_threshold:
                    x1 = min(merged_box[0], box[0])
                    y1 = min(merged_box[1], box[1])
                    x2 = max(merged_box[0] + merged_box[2], box[0] + box[2])
                    y2 = max(merged_box[1] + merged_box[3], box[1] + box[3])

                    merged_boxes[i] = [x1, y1, x2 - x1, y2 - y1]
                    should_merge = True
                    break

            if not should_merge:
                merged_boxes.append(box)

        return merged_boxes

    def process_frame(self, frame, user_name):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(img)

        boxes = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                boxes.append([x, y, w, h])

            merged_boxes = self.merge_boxes(boxes)

            for box in merged_boxes:
                p1 = (box[0], box[1])
                p2 = (box[0] + box[2], box[1] + box[3])
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

                if cv2.waitKey(1) & 0xFF == ord('c'):
                    self.save_face(frame, p1, p2, user_name)

        return frame

    def save_face(self, frame, p1, p2, user_name):
        raw_data_path = os.path.join(self.ingestion_config.raw_data_path, user_name)
        os.makedirs(raw_data_path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        file_path = os.path.join(raw_data_path, f'captured_face-{timestamp}.jpg')
        roi = frame[p1[1]:p2[1], p1[0]:p2[0]]
        cv2.imwrite(file_path, roi)
        logging.info(f"Saved captured face to {file_path}")

    def initiate_data_ingestion(self):
        logging.info('Initiated Data Ingestion')
        try:
            user_name = input('Enter User Name: ')

            vid = cv2.VideoCapture(0)
            if not vid.isOpened():
                logging.error("Error: Could not open webcam.")
                return

            while True:
                ret, frame = vid.read()
                if not ret:
                    logging.error("Error: Could not read frame from webcam.")
                    break

                frame = self.process_frame(frame, user_name)
                cv2.imshow('frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            vid.release()
            cv2.destroyAllWindows()

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
