import glob
import numpy as np
from Util import Util
import cv2
import time

# Load model
net = cv2.dnn.readNetFromDarknet('../yolov4-tiny-custom.cfg', '../yolov4-tiny-custom_final.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

util = Util()

def getBBfromFile(text, dw, dh):
    boxes = []
    lines = text.splitlines()
    for line in lines:
        _, x, y, w, h = map(float, line.split(' '))
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1
        boxes.append( [ l, t, r, b ] )
    return boxes

def renderBB(box,color):
    l, t, r, b = box
    cv2.rectangle(img, (l, t), (r, b), color, 2)

def PerformYolo(frame):

    (H, W) = frame.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i- 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.4:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    detected_cars = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            detected_cars.append([x,y,x+w,y+h])

    return detected_cars


# Testing - - - - - - - - - - - - - - - - - - - - - - -

images = glob.glob("../testing_dataset/*.png")
labels = glob.glob("../testing_dataset/*.txt")

img_num = 0
total_cars = 0
correct_predictions = 0
iou_for_correct_predictions = 0
prediction_time = []

# PRESS Q TO GO TROUGH ALL IMAGES! <! - - - - -

img_iterator = 0
while img_iterator < len(images):

    # Get image
    img = cv2.imread(images[img_iterator])
    dh, dw, _ = img.shape

    # Get groundtruth bounding box
    bbfile = open(labels[img_iterator], "r")
    bb = bbfile.read()
    truth_boxes = getBBfromFile(bb, dw, dh)

    # Get pred boxes from yolo
    start = time.time()
    pred_boxes = PerformYolo(img)
    end = time.time()
    prediction_time.append( end-start )

    img_num += 1
    total_cars += len(truth_boxes)

    # Calculate iou
    for box_gt in truth_boxes:

        best_iou_value = 0
        index = 0

        j = 0
        while j < len(pred_boxes):
            iou_value = util.bb_iou( box_gt, pred_boxes[j] )
            if iou_value > best_iou_value:
                index = j
                best_iou_value = iou_value
            j += 1

        # if high iou
        if best_iou_value > 0.5: # Match
            correct_predictions += 1
            iou_for_correct_predictions += best_iou_value

    # Render bounding boxes
    for box in truth_boxes:
        renderBB(box, (0,255,0))
    for box in pred_boxes:
        renderBB(box, (0,0,255))

    cv2.imshow('frame', img)
    cv2.waitKey(0)
    img_iterator += 1
cv2.destroyAllWindows()


print( "Average predicition time:", str(round(np.average(prediction_time),4)) + " seconds"  )
print( "Accuracy:", float(correct_predictions)/float(total_cars) )
print( "Average iou for correct predictions:",float(iou_for_correct_predictions)/float(correct_predictions))