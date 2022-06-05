# Traffic Monitoring System for Low-end Devices


<img src="https://serooshsaba.github.io/monitor.gif" width="100%">

<br/><br/>

One problem with neural networks for object-detection is that they require a GPU to run in real-time. But
this is not possible for many embedded device that do not have a built-in GPU. These devices then
need to use the CPU, which is not optimal for running the neural network in real-time. This basically
makes these devices useless for real-time traffic monitoring. Another problem is that even if there is a
sufficiently fast GPU built-in, the computation will have a big impact on the batteries of these devices
if used.

Because of this issue i created a traffic monitoring system, which does not rely on neural
network driven object-detection for tracking. Neural networks are still used sparingly to determine
what the kind of vehicle is being tracked, while unloading the task of tracking to other less
computationally expensive methods within imageanalysis. This way there is be less usage for object
detection, and therefore more low-end embedded devices will be able to be used, to improve traffic
safety.

## Steps for movement detection

### 1. Convert to grayscale and blur

Grayscaling removes the red, green and blue color channels of the image, which makes the image
easier and faster to process, and also enables the use of specific functions. Blurring reduces small movements that result in noise and errors.

### 2. Find the difference between the current and previous frame

The difference
means we subtract the values of corresponding pixels in each image.
Pixels that have not changed between the two frames will have the
same value, and therefore result to 0 after subtraction. But for pixels
that have changed because of movement, will result in a value > 0.

### 3. Apply a treshold filter

This process is then followed by tresholding, which only pass
trough pixels that have experianced a certain amount of change in
values, which helps to remove noise. The images that pass the
threshold are returned as fully white pixels with the value 255.

### 4. Apply a mask to isolate relevant sections

Some areas of a videostream can be a source of noise and error when strong winds 
are present. This needs to be filtered out, which is very easily removed with 
the use of a custom mask, applied with a bitwise-and operator between the mask
and the previous threshold image. 

### 5. Dilute the treshold values

The next step is to dilute the white pixels of the image after
masking. This expands the area of the objects of movement, which
leads to better covering bounding-boxes.

### 6. Extract objects

We first find the contours that are
around the white pixels of the diluted image. After this the contours are filtered so that only the ones
that are sufficiently large in area are passed futher in the process. Then we convert these contours into
bounding boxes. Which represent areas of sufficient movement, which again represents the vehicles
on the road. Trough this simple method the system can easily detect movements in real-time

### 7: Track trough frames

To do this we find the
corresponding bounding boxes from frame to frame. We know
that two bounding boxes from two frames are the same when
there is a high value of intersection over union (IoU) between
them. In this way we can track the same bounding boxes through
multiple frames.

### 8: Classify

With the trained object detector we can find out what kind of vehicles are present
in frame.

## Misc:
- YoloV4-Tiny for object detection
- OpenCV2 with Python 3 for logic and image manipulation.

Full video for demonstration: https://www.youtube.com/watch?v=Vd0UMd8A0Lk&t=2s

