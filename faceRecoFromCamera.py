import csv
import logging
import os.path
import pstats

import cv2
import numpy as np
import time
import pandas as pd

from trtFacenet import TRTFacenet
from trtRetinaFace import TRTRetinaFace

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class FaceRecognizer(object):
    def __init__(self, retinaFace_path, facenet_path, video_path, threshold):
        self.feature_csv = 'data' + os.sep + 'feature.csv'
        self.retinaFace = TRTRetinaFace(retinaFace_path)
        self.facenet = TRTFacenet(facenet_path)
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)
        self.font = cv2.FONT_HERSHEY_DUPLEX

        self.threshold = threshold

    def judge_face_number(self, faceList):
        if len(faceList) == 0 or len(faceList) == 1:
            return len(faceList), "%s face was detected" % len(faceList)
        else:
            return len(faceList), "%s faces were detected" % len(faceList)

    def get_features_from_csv(self):
        if not os.path.exists(self.feature_csv):
            return [], []

        else:
            csv_pd = pd.read_csv(self.feature_csv, header=None)
            feature_list = []
            id_list = []
            for i in range(csv_pd.shape[0]):
                feature_list.append(csv_pd.iloc[i][1:129])
                id_list.append(csv_pd.iloc[i][0])

            return feature_list, id_list

    @staticmethod
    def cal_distance(feature_database, feature_detected):
        l1 = np.linalg.norm(feature_database - feature_detected, axis=0)
        return l1

    def find_person(self, feature_list, id_list, feature_detected):
        for i, feature in enumerate(feature_list):
            feature = feature.to_numpy()
            feature_detected = np.squeeze(feature_detected)
            if self.cal_distance(feature, feature_detected) < self.threshold:
                return id_list[i]
        return -1

    def draw_note(self, image, numberInfo, dets, person_ids, fps):
        cv2.putText(image, "Face Recognizer", (20, 40), self.font, 1, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, numberInfo, (20, 80), self.font, 1, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "FPS: " + "{:.2f}".format(fps), (image.shape[1] - 200, 50), self.font, 1, (255, 0, 255), 1,
                    cv2.LINE_AA)
        for i, b in enumerate(dets):
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(image, text, (cx, cy), self.font, 0.5, (255, 255, 255))
            cv2.putText(image, str(person_ids[i]), (cx, cy + 12), self.font, 0.5, (255, 255, 255))
        return image

    def process(self):
        feature_list, id_list = self.get_features_from_csv()
        while self.capture.isOpened():
            t1 = time.time()
            ret, frame = self.capture.read()
            frame = cv2.resize(frame, (640, 480))
            img_raw = frame
            frame = frame.astype(np.float32)
            res, faceList = self.retinaFace.detect(frame)
            length, numbInfo = self.judge_face_number(faceList)
            person_ids = []
            if len(faceList) != 0:
                for face in faceList:
                    feature_detected = self.facenet.detect(face)
                    person_ids.append(self.find_person(feature_list, id_list, feature_detected[0]))

            fps = 1 / (time.time() - t1)
            img = self.draw_note(img_raw, numbInfo, res, person_ids, fps)
            cv2.waitKey(1)
            cv2.imshow("win", img)


if __name__ == '__main__':
    retinaFace_engine_path = 'model/retinaFace.trt'
    facenet_engine_path = 'model/facenet_op10.trt'
    faceReco = FaceRecognizer(retinaFace_engine_path, facenet_engine_path, 0, 0.6)
    faceReco.process()
