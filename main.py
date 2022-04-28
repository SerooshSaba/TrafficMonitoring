from Tracker import Tracker
import cv2
import numpy as np

# Load object detection model
net = cv2.dnn.readNetFromDarknet('yolov4-tiny-custom.cfg', 'yolov4-tiny-custom_final.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

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

tracker = Tracker()

FRAME = 1
FRAMETIME = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Load mask image
mask = cv2.imread('mask.png',0)
mask = 255 - mask
mask = np.array(mask,dtype='uint8')

PREVIOUS_FRAME = None
cap = cv2.VideoCapture('video3.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 0 )


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        if PREVIOUS_FRAME is None: # If first frame, then just process frame and save it as previous frame and continue
            prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            PREVIOUS_FRAME = prepared_frame
        else:

            prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff_frame = cv2.absdiff(src1=PREVIOUS_FRAME, src2=prepared_frame)
            thresh_frame = cv2.threshold(src=diff_frame, thresh=50, maxval=255, type=cv2.THRESH_BINARY)[1]
            thresh_frame = cv2.bitwise_and(thresh_frame, thresh_frame, mask=mask)
            dilated = cv2.dilate(thresh_frame, None, iterations=3)
            contours, _ = cv2.findContours(image=dilated, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)


            # Filter boxes
            boxes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    if w * h < 25000:
                        boxes.append( [x,y,x+w,y+h] )

            # Track boxes
            tracker.cleanup_tracker()
            tracker.processInputs(boxes)

            # If there are unlabelled objects currently being tracked
            if tracker.unlabeled_object_in_frame():
                cars_in_frame = PerformYolo(frame)
                #tracker.drawYolo(frame=frame,cars=cars_in_frame)
                tracker.setlabel(cars_in_frame)
            
            tracker.drawOn(frame=frame, font=FONT)
            cv2.imshow('frame', frame)

            FRAME += 1
            PREVIOUS_FRAME = prepared_frame
        if cv2.waitKey(FRAMETIME) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()