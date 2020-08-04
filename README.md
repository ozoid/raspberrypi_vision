# raspberrypi_vision
Coral TPU, Haar, AprilTag, Caffe, Object Detection with ROS publisher and Flask MJpeg Stream

A multithreaded solution for object detection, ROS publishing and output to MJPEG Stream from a raspberry Pi 4 with Coral Edge TPU USB Device.

No screen output - via http port 8000/video_feed (mjpeg stream)

Models not included - available freely elsewhere

Requires opencv + imutils

Custom ROS Message for data - publishes on /vision_detect topic
