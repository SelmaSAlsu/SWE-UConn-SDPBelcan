# https://youtu.be/oXlwWbU8l2o?feature=shared
# % pip install opencv-contrib-python
# % pip install caer
import os
import cv2 as cv
import numpy as np

# only works if you have GNU WGET installed. otherwise, do it manually lol
os.system("wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights") if not os.path.exists("yolov4-tiny.weights") else 0
os.system("wget https://github.com/AlexeyAB/darknet/raw/master/cfg/yolov4-tiny.cfg") if not os.path.exists("yolov4-tiny.cfg") else 0
os.system("wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names") if not os.path.exists("coco.names") else 0

net = cv.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
cap = cv.VideoCapture(0)
camera_active = False
confidence_threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if ret and not camera_active:
        camera_active = True
        print("camera is active and recording")

    blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x = int(obj[0] * frame.shape[1])
                center_y = int(obj[1] * frame.shape[0])
                w = int(obj[2] * frame.shape[1])
                h = int(obj[3] * frame.shape[0])

                x = center_x - w // 2
                y = center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            if label == "dog":
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame, f"{label} {confidences[i]:.2f}", (x, y - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv.imshow("Dog Tracker", frame)

    # HIT THE "Q" KEY TO END THE TRACKING PROGRAM
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Q pressed, quitting program")
        break


cap.release()
cv.destroyAllWindows()