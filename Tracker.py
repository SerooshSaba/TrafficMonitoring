import random
import string
import cv2
from Util import Util

TRACKING_TRESHOLD = 15

class TrackingObject:

    classes = [ "Car", "Vehicle" ]

    def __init__(self,box):
        self.name = str(''.join(random.choices(string.ascii_uppercase + string.digits, k=3)))
        self.label = None
        self.frames_since_last_updated = 0
        self.position = box

        # Positions of vehicle trough frames
        self.path = [ (int((box[0]+box[2])/2),int((box[1]+box[3])/2)) ]

    def update(self,box):
        self.position = box
        self.path.append( (int((box[0]+box[2])/2), int((box[1]+box[3])/2)) )

    def getCopy(self):
        return self.position.copy()

    def render(self,frame,font):
        pt1 = self.position[:2]
        pt2 = self.position[2:]

        # Draw updated bounding boxes and ignore old boxes
        if self.frames_since_last_updated <= 1 and len(self.path) > TRACKING_TRESHOLD:
            # Draw bounding box of vehicle
            cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=2)
            if self.label is not None:

                if self.label == 0:
                    text_size, _ = cv2.getTextSize(self.classes[self.label] + "-" + self.name, font, 0.5, 1)
                    text_w, text_h = text_size
                    cv2.rectangle(frame, (pt1[0],pt1[1]+2), (pt1[0] + text_w, pt1[1] - text_h), (0,0,0), -1)
                    cv2.putText(frame, self.classes[self.label] + "-" + self.name, (pt1[0]+1,pt1[1]), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
                else:
                    text_size, _ = cv2.getTextSize(self.classes[self.label] + "-" + self.name, font, 0.5, 1)
                    text_w, text_h = text_size
                    cv2.rectangle(frame, (pt1[0], pt1[1] + 2), (pt1[0] + text_w, pt1[1] - text_h), (0, 0, 0), -1)
                    cv2.putText(frame, self.classes[self.label] + "-" + self.name, (pt1[0] + 1, pt1[1]), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


class Tracker:

    FIRST_FRAME = True
    util = Util()

    def __init__(self):
        self.objects = []

    def processInputs(self,boxes):

        if len(boxes) > 0:
            # First frame has no previous bb to compare to, so we just convert all bb to vehicles
            if self.FIRST_FRAME:
                self.FIRST_FRAME = False
                for box in boxes:
                    self.objects.append(TrackingObject(box))

            # After first frame we need to update vehicle positions, and check for new ones that pop into perspective
            else:

                no_iou = []
                # For each input boxes, find the best match with the registered vehicles
                for box in boxes:
                    best_iou = 0
                    index = 0

                    # Find best iou match
                    i = 0
                    while i < len(self.objects):
                        iou_val = self.util.bb_iou(self.objects[i].position, box)
                        if iou_val > best_iou:
                            best_iou = iou_val
                            index = i
                        i += 1

                    # Update vehicle if big iou match
                    if best_iou > 0.5:
                        self.objects[index].update(box)
                        self.objects[index].frames_since_last_updated = 0
                    else:
                        no_iou.append(box)

                # Boxes that have 0 iou get added as new vehicles
                for box in no_iou:
                    self.objects.append(TrackingObject(box))


    def cleanup_tracker(self):

        # Update the age of bounding boxes
        i = 0
        while i < len(self.objects):
            self.objects[i].frames_since_last_updated += 1
            i += 1

        # Delete bounding boxes that are stuck and not updated
        i = len(self.objects) - 1
        while i >= 0:
            if self.objects[i].frames_since_last_updated > 5:
                del self.objects[i]
            i -= 1

    def saveObjectPositions(self):
        positions = []
        for object in self.objects:
            positions.append( object.getCopy() )
        return positions

    def unlabeled_object_in_frame(self):
        for object in self.objects:
            if len(object.path) > TRACKING_TRESHOLD and object.label is None:
                return True
        return False


    def setlabel(self,cars):

        if len(self.objects) == len(self.objects):

            i = 0
            while i < len(self.objects):

                if self.objects[i].label == None:
                    best_iou = 0

                    for box in cars:
                        iou_val = self.util.bb_iou(self.objects[i].position, box)
                        if iou_val > 0.4:
                            best_iou = iou_val
                            break
                        elif iou_val > best_iou:
                            best_iou = iou_val

                    if best_iou > 0.3:
                        self.objects[i].label = 0
                    else:
                        self.objects[i].label = 1
                i += 1

    def drawYolo(self,frame,cars):
        for box in cars:
            pt1 = box[:2]
            pt2 = box[2:]
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 5)

    def drawOn(self, frame, font):
        for object in self.objects:
            # Render driving path of vehicle
            path = object.path
            color = (0,0,255) if len(path) < TRACKING_TRESHOLD else (0, 255, 0)
            for i in range(0, len(path)-1):
                pt1 = path[i]
                pt2 = path[i+1]
                cv2.line(frame, pt1, pt2, color, 2)
            # Render vehicle box
            object.render(frame, font)