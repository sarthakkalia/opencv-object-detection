import cv2

import numpy as np 
import time

# np.random.seed(20)    #if this is used then single color generate
class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, '__Background__')

        self.colorList=np.random.uniform(low=0,high=255,size=(len(self.classesList), 3))

        # print(self.classesList)
    def calculate_distance(self, bbox):
        actual_object_height_in_meters = 0.1  # Example: object height is 10 cm
        focal_length = 1000  # Example: focal length is 1000 pixels
        object_height_in_pixels = bbox[3]

        distance = (actual_object_height_in_meters * focal_length) / object_height_in_pixels

        return distance


    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Error opening file..")
            return

        (success, image) = cap.read()

        startTime = 0

        while success:
            currentTime = time.time()
            fps = 1 / (currentTime - startTime)
            startTime = currentTime

            classsLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.4)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidences = confidences[np.squeeze(bboxIdx[i])]
                    classsLabelID = np.squeeze(classsLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classsLabelID]
                    classColor = [int(c) for c in self.colorList[classsLabelID]]

                    displayText = "{}:{:.2f}".format(classLabel, classConfidences)

                    x, y, w, h = bbox
                    distance = self.calculate_distance(bbox)

                    # Display distance above the respective object
                    distance_text = "Distance: {:.2f} meters".format(distance)
                    cv2.putText(image, distance_text, (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

                    cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=1)
                    cv2.putText(image, displayText, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

            cv2.putText(image, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow("Result", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (success, image) = cap.read()
        cv2.destroyAllWindows()
