import csv
import logging
import os.path
import pstats

import cv2
import numpy as np
import time

from trtFacenet import TRTFacenet
from trtRetinaFace import TRTRetinaFace


class FaceRegister(object):
    def __init__(self, retinaFace_path, facenet_path, video_path):
        self.retinaFace = TRTRetinaFace(retinaFace_path)
        self.facenet = TRTFacenet(facenet_path)
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)
        self.font = cv2.FONT_HERSHEY_DUPLEX

        self.path_photos_from_camera = 'data' + os.sep + 'photo'
        self.feature_csv = 'data' + os.sep + 'feature.csv'
        self.existing_face_cnt = 0
        self.ss_cnt = 0
        self.press_n_flag = 0

    def judge_face_number(self, faceList):
        if len(faceList) == 0:
            return "NO face was detected."
        elif len(faceList) == 1:
            return "1 face was detected."
        else:
            return "TOO MANY faces were detected."

    def draw_note(self, image, numberInfo, dets, fps):
        cv2.putText(image, "Face Register", (20, 40), self.font, 1, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, numberInfo, (20, 80), self.font, 1, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "FPS: " + "{:.2f}".format(fps), (image.shape[1] - 200, 50), self.font, 1, (255, 0, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(image, "Press \'n\' to add a new person", (20, image.shape[0] - 80), self.font, 0.5, (255, 0, 255),
                    1, cv2.LINE_AA)
        cv2.putText(image, "Press \'s\' to save photo", (20, image.shape[0] - 50), self.font, 0.5, (255, 0, 255),
                    1, cv2.LINE_AA)
        cv2.putText(image, "Press \'q\' to quit", (20, image.shape[0] - 20), self.font, 0.5, (255, 0, 255),
                    1, cv2.LINE_AA)
        for b in dets:
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(image, text, (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
        return image

    def process(self):
        if not os.path.exists('data'):
            os.mkdir('data')
        if not os.path.exists(self.path_photos_from_camera):
            os.mkdir(self.path_photos_from_camera)

        if not os.path.exists(self.feature_csv):
            with open(self.feature_csv, 'w'):
                pass

        self.existing_face_cnt = len(os.listdir(self.path_photos_from_camera))

        while self.capture.isOpened():
            t1 = time.time()
            ret, frame = self.capture.read()
            frame = cv2.resize(frame, (640, 480))
            img_raw = frame
            frame = frame.astype(np.float32)
            res, faceList = self.retinaFace.detect(frame)
            numberInfo = self.judge_face_number(faceList)
            keyboard = cv2.waitKey(1)
            fps = 1 / (time.time() - t1)

            if keyboard == ord('n'):
                logging.info("Add a new person")
                self.existing_face_cnt += 1
                photo_path = self.path_photos_from_camera + os.sep + str(self.existing_face_cnt + 1)
                os.mkdir(photo_path)

                self.ss_cnt = 0
                self.press_n_flag = 1

            if len(faceList) == 1:
                if keyboard == ord('s'):
                    logging.info("s button was pressed")
                    if self.press_n_flag:
                        self.ss_cnt += 1
                        faceFeature = self.facenet.detect(faceList[0])
                        cv2.imwrite(photo_path + os.sep + str(self.ss_cnt) + '.jpg', faceList[0])
                        with open(self.feature_csv, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            faceFeature[0] = np.r_[self.existing_face_cnt, faceFeature[0]]
                            writer.writerow(np.squeeze(faceFeature[0]))

            if keyboard == ord('q'):
                break
            img_raw = self.draw_note(img_raw, numberInfo, res, fps)
            cv2.imshow("win", img_raw)


if __name__ == "__main__":
    retinaFace_engine_path = 'model/retinaFace.trt'
    facenet_engine_path = 'model/facenet_op10.trt'
    facereg = FaceRegister(retinaFace_engine_path, facenet_engine_path, 0)
    facereg.process()
