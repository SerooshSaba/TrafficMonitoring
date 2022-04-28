from Util import Util
import cv2
import numpy as np

util = Util()

RENDER_AREA = True
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Load mask image
mask = cv2.imread('mask.png',0)
mask = 255 - mask
mask = np.array(mask,dtype='uint8')

scale_percent = 50
width = int(mask.shape[1] * scale_percent / 100)
height = int(mask.shape[0] * scale_percent / 100)
dim = (width, height)

# Load image n and n-1
im1 = cv2.imread('1.jpg')
im2 = cv2.imread('2.jpg')

# resize image
mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)
im1 = cv2.resize(im1, dim, interpolation=cv2.INTER_AREA)
im2 = cv2.resize(im2, dim, interpolation=cv2.INTER_AREA)

# To grayscale
im1gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
#cv2.imshow('im1gray', im1gray)
#cv2.imshow('im2gray', im2gray)

# Find difference between images
diff_frame = cv2.absdiff(src1=im1gray, src2=im2gray)
#cv2.imshow('difference', diff_frame)

# Treshold the image
thresh_frame = cv2.threshold(src=diff_frame, thresh=40, maxval=255, type=cv2.THRESH_BINARY)[1]
#cv2.imshow('treshold', thresh_frame)

# Apply mask
thresh_frame = cv2.bitwise_and(thresh_frame, thresh_frame, mask=mask)
#cv2.imshow('mask', thresh_frame)

# Dilate image
dilated = cv2.dilate(thresh_frame, None, iterations=2)
#cv2.imshow('dilate', dilated)

# Find contours
contours, _ = cv2.findContours(image=dilated, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image=im2, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
#cv2.imshow('contours', im2)

boxes = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 800:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w * h < 20000:
            boxes.append( [x,y,w,h,area] )

# Render final bounding boxes
for box in boxes:
    (x,y,w,h,area) = box
    cv2.rectangle(img=im2, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

cv2.imshow('contours', im2)

cv2.waitKey(0)

"""
prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
thresh_frame = cv2.threshold(src=diff_frame, thresh=40, maxval=255, type=cv2.THRESH_BINARY)[1]
thresh_frame = cv2.bitwise_and(thresh_frame, thresh_frame, mask=mask)
dilated = cv2.dilate(thresh_frame, None, iterations=2)
contours, _ = cv2.findContours(image=dilated, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(image=frame, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

# Filter bounding boxes
boxes = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w * h < 20000:
            boxes.append( [x,y,w,h,area] )

# Render final bounding boxes
for box in boxes:
    (x,y,w,h,area) = box
    cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

# Render area of bounding box
if RENDER_AREA:
    cv2.rectangle(img=frame, pt1=(x, y - 13), pt2=(x + w, y + int(h*0.1)), color=(0, 0, 0), thickness=-1)
    frame = cv2.putText(frame, str(area), (x,y), FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

#cv2.imshow('frame', frame)
cv2.imshow('diff_frame', diff_frame)
previous_frame = prepared_frame
"""