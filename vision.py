#!/usr/bin/env python3
# Author: Steve Lawrence - Ozoid Robotics 2020 - Ozoid Ltd.
# 
import os
import time
import picamera
import numpy as np
import cv2
import argparse
import io
import time
import imutils
import itertools
import rospy
from threading import Thread, Lock
from flask import Flask, Response
from robot5.msg import vision_detect
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw
from imutils.video import VideoStream
from dt_apriltags import Detector


class Vision:
    coral_model_file = "/home/pi/catkin_ws/src/robot5/models/mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
    coral_labels_file = "/home/pi/catkin_ws/src/robot5/models/mobilenet_ssd_v2/coco_labels.txt"
    coral_confidence = 0.3
    caffe_model_file = "/home/pi/catkin_ws/src/robot5/models/res10_300x300_ssd_iter_140000.caffemodel"
    caffe_confidence = 0.5
    caffe_prototxt = "/home/pi/catkin_ws/src/robot5/models/deploy.prototxt.txt"
    face_cascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_eye.xml')

    APRILWIDTH = 172
    FOCALLENGTH = 0.304
    def __init__(self):
        self.coral_model = {}
        self.coral_labels = {}
        self.caffe_model = {}
        self.at_detector = {}
        self.videoStream = {}
        self.status = None
        self.captureFrame = None
        self.visionFrame = None
        self.thread = Thread(target=self.frameUpdate,args=())
        self.thread.daemon = True
        self.flaskThread = Thread(target=self.runFlask)
        self.flaskThread.daemon = True
        self.frameLock = Lock()
        print("[INFO] Initialising ROS...")
        #self.pub = rospy.Publisher(name='vision_detect',subscriber_listener=vision_detect,queue_size=5,data_class=vision_detect)
        self.pub = rospy.Publisher('/vision_detect',vision_detect)
        rospy.init_node('robot5_vision',anonymous=False)
        
        for row in open(self.coral_labels_file):
	        (classID, label) = row.strip().split(maxsplit=1)
	        self.coral_labels[int(classID)] = label.strip()

        print("[INFO] loading Coral model...")
        self.coral_model = DetectionEngine(self.coral_model_file)
        #print("[INFO] loading Caffe model...")
        #self.caffe_model = cv2.dnn.readNetFromCaffe(self.caffe_prototxt, self.caffe_model_file)
        self.at_detector = Detector(families='tag36h11',
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0)
        print("[INFO] Running Flask...")
        self.app = Flask(__name__)
        self.add_routes()
        self.flaskThread.start()
        print("[INFO] starting video stream...")
        self.videoStream = VideoStream(src=0,usePiCamera=True).start()
        time.sleep(2.0) # warmup
        self.captureFrame = self.videoStream.read()
        self.visionFrame = self.captureFrame
        self.thread.start()
        time.sleep(0.5) # get first few
        srun = True
        print("[INFO] running frames...")
        while srun:
          srun = self.doFrame()
        cv2.destroyAllWindows()
        self.videoStream.stop()

    def frameUpdate(self):
        while True:
          self.captureFrame = self.videoStream.read()
          time.sleep(0.03)
 
    def gen(self):
        while True:
            if self.visionFrame is not None:
              bout = b"".join([b'--frame\r\nContent-Type: image/jpeg\r\n\r\n', self.visionFrame,b'\r\n'])
              yield (bout)
            else:
              return ""

    def add_routes(self):
        @self.app.route("/word/<word>")
        def some_route(word):
            self.testout()
            return "At some route:"+word

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

    def testout(self):
        print("tested")
        pass

    def runFlask(self):
        self.app.run(debug=False, use_reloader=False,host='0.0.0.0', port=8000)

    def coralDetect(self,frame,orig):
      start1 = time.time()
      results = self.coral_model.detect_with_image(frame, threshold=self.coral_confidence,keep_aspect_ratio=True, relative_coord=False)
      fcount = 0
      points = []
      for r in results:
        box = r.bounding_box.flatten().astype("int")
       	(startX, startY, endX, endY) = box
       	label = self.coral_labels[r.label_id]
       	cv2.rectangle(orig, (startX, startY), (endX, endY),	(0, 255, 0), 2)
       	y = startY - 15 if startY - 15 > 15 else startY + 15
        cx = startX + (endX - startX/2)
        cy = startY + (endY - startY/2)
        #points.append({"type":"coral","num":fcount,"x":cx,"y":cy,"label":label,"score":r.score,"time":time.time() })
        points.append(["coral",fcount,int(cx),int(cy),label,int(r.score*100),rospy.Time.now()])
        fcount +=1
       	text = "{}: {:.2f}%".format(label, r.score * 100)
       	cv2.putText(orig, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      end1 = time.time()
      #print("#1:",end1-start1)
      return orig,points

    def caffeDetect(self,frame,orig):
      start2 = time.time()
      (h, w) = frame.shape[:2]
      blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
      self.caffe_model.setInput(blob)
      detections = self.caffe_model.forward()
      fcount = 0
      points = []
      for i in range(0, detections.shape[2]):
          confidence = detections[0, 0, i, 2]
          if confidence < self.caffe_confidence:
              continue
          box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
          (startX, startY, endX, endY) = box.astype("int")
          text = "{:.2f}%".format(confidence * 100)
          y = startY - 10 if startY - 10 > 10 else startY + 10
          cx = startX + (endX - startX/2)
          cy = startY + (endY - startY/2)
          #points.append({"type":"caffe","num":fcount,"x":cx,"y":cy,"score":confidence,"time":time.time()})
          points.append(["caffe",fcount,int(cx),int(cy),"",int(confidence*10),rospy.Time.now()])
          fcount +=1
          cv2.rectangle(orig, (startX, startY), (endX, endY),(0, 0, 255), 2)
          cv2.putText(orig, text, (startX, y),	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
      end2 = time.time()
      #print("#2:",end2-start2)
      return orig,points

    def aprilDetect(self,grey,orig):
      start3 = time.time()
      Xarray = [338.563277422543, 0.0, 336.45495347495824, 0.0, 338.939280638548, 230.486982216255, 0.0, 0.0, 1.0]
      camMatrix = np.array(Xarray).reshape((3,3))
      params = (camMatrix[0,0],camMatrix[1,1],camMatrix[0,2],camMatrix[1,2])
      tags = self.at_detector.detect(grey,True,params,0.065)
      fcount = 0
      points =[]
      for tag in tags:
        pwb = tag.corners[2][1] - tag.corners[0][1]
        pwt = tag.corners[3][1] - tag.corners[1][1]
        pwy = (pwb+pwt)/2
        pwl = tag.corners[3][1] - tag.corners[0][1]
        pwr = tag.corners[2][1] - tag.corners[1][1]
        pwx = (pwl+pwr)/2
        dist = self.distanceToCamera(self.APRILWIDTH,(pwx))
        #print(dist)
        #points.append({"type":"april","num":fcount,"x":pwx,"y":pwy,"label":tag.id,"score":dist,"time":time.time()})
        points.append(["april",fcount,int(pwl + (pwx/2)),int(pwb + (pwy/2)),str(tag.tag_id)+"|"+str(dist),0,rospy.Time.now()])
        fcount += 1
        cv2.putText(orig, str(dist), (int(pwx), int(pwy)),	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        for idx in range(len(tag.corners)):
            cv2.line(orig,tuple(tag.corners[idx-1,:].astype(int)),tuple(tag.corners[idx,:].astype(int)),(0,255,0))
      end3 = time.time()
      #print("#3:",end3-start3)
      return orig,points

    def haarDetect(self,grey,orig):
        start4 = time.time()
        faces = self.face_cascade.detectMultiScale(grey,1.3,5)
        fcount=0
        points =[]
        for(x,y,w,h) in faces:
          #points.append({"type":"haar","num":fcount,"x":x+(w/2),"y":y+(h/2),"time":time.time()})
          points.append(["haar",fcount,int(x+(w/2)),int(y+(h/2)),"",0,rospy.Time.now()])
          orig = cv2.rectangle(orig,(x,y),(x+w,y+h),(255,255,0),2)
          roi_gray = grey[y:y+h, x:x+w]
          roi_color = orig[y:y+h, x:x+w]
          fcount += 1
          eyes = self.eye_cascade.detectMultiScale(roi_gray)
          for (ex,ey,ew,eh) in eyes:
              cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        end4 = time.time()
        #print("#4:",end4-start4)
        return orig,points

    def doFrame(self):
        frame = self.captureFrame
        if frame is None:
          return False
        frame = imutils.resize(frame, width=500)
        orig = frame.copy()
        grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        outframe,cpoints = self.coralDetect(frame,orig)
        outframe,apoints = self.aprilDetect(grey,outframe)
        outframe,hpoints = self.haarDetect(grey,outframe)
        #outframe,fpoints = self.caffeDetect(orig,outframe)
        points = list(itertools.chain(cpoints,apoints,hpoints))
        for p in points:
          self.pub.publish(p[0],p[1],p[2],p[3],p[4],p[5],p[6])
          pass
        ret, self.visionFrame = cv2.imencode('.jpg', outframe)
        #self.visionFrame = outframe
        #cv2.imshow("Frame", outframe)
        #key = cv2.waitKey(1) & 0xFF
        #if key == ord("q"):
        #  return False
        return True

    def distanceToCamera(self,actWidth,perWidth):
        return ((actWidth*self.FOCALLENGTH)/perWidth)*2

if __name__ == '__main__':
  v = Vision()
