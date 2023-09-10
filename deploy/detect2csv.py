import cv2
import numpy as np
import csv
# "C:\\Users\\jagan\\Desktop\\others\\test2.mp4"
cap = cv2.VideoCapture("test2.mp4")
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
# f = open('csv_file/first.csv', 'w', encoding='UTF8')

classesFile = 'data/yolo.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)
# print(len(classNames))

modelConfiguration = 'cfg/yolov3_custom_train.cfg'
modelWeights = 'yolov3_custom_train_1000.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
k = 0
# global frame
f = open('csv_file/first.csv', 'a', encoding='UTF8', newline="")
writer = csv.writer(f, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
frame = 0
count1 = 1


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    global frame
    global count1
    count = ["frame ="]
    count1 = 0

    frame += 1
    count.append(frame)
    writer.writerow(count)
    # print(frame)

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    # frame = frame + 1
    # head = "cars"
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, "CAR", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        if cv2.rectangle:

            count.append("cars =")
            count1 += 1
            count.append(count1)
    writer.writerow(count)

    '''if cv2.rectangle:
            count.append("cars =")
            count1 = 1
            count.append(count1)
            writer.writerow(count)
            # writer = csv.writer(f)

         elif cv2.rectangle == 2:
            count.append("cars =")
            count1 = 2
            count.append(count1)
            writer.writerow(count)
            print(count)
        #   print(count1)
        #else:

        #    count.append("cars =")
        #    count1 = 3
        #    count.append(count1)
        #    writer.writerow(count)
            # print(count)
            # print("cars=", count1)

'''

# f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%'

while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    # print(layerNames)
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    # print(net.getUnconnectedOutLayers())

    outputs = net.forward(outputNames)
    # print(len(outputs))
    # print(type(outputs[0]))
    # print(outputs[0].shape) (300, 85)
    # print(outputs[1].shape) (1200, 85)
    # print(outputs[2].shape) (4800, 85)
    # print(outputs[0][0])

    findObjects(outputs, img)

    cv2.imshow('Image', img)
    cv2.waitKey(1)