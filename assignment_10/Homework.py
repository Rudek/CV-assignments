### Step 1
#Decide what video you are going to use for this homework, select an object and generate the template. You can use any video you want (your own, from Youtube, etc.)
#and track any object you want (e.g. a car, a pedestrian, etc.).

import cv2 as cv
import matplotlib.pyplot as plt


### Step 2
# Set up tracker
tracker_types = ['MIL', 'KCF', 'CSRT']
tracker_type = tracker_types[2]


if tracker_type == 'MIL':
    tracker = cv.TrackerMIL_create()

if tracker_type == 'KCF':
    tracker = cv.TrackerKCF_create()

if tracker_type == "CSRT":
    tracker = cv.TrackerCSRT_create()

videos = ['traffic_2', 'traffic_3']
video = videos[1]

# Object square on video traffic_2.mp4
if video == 'traffic_2':
  x1, y1 = 210, 173
  x2, y2 = 260, 203

if video == 'traffic_3':
  # Object square on video traffic_3.mp4
  x1, y1 = 166, 173
  x2, y2 = 190, 192

width = x2 - x1
height = y2 - y1


cap = cv.VideoCapture(f"/Users/maxim.s.rudenko/Self-Study/Computer Vision/Assignments/Homeworks/assignment_10/{video}.mp4")

if not cap.isOpened():
  print("Cannot open camera")
  exit()

ret, frame = cap.read()

if not ret:
  print("Can't receive frame (stream end?). Exiting ...")
  exit()

bbox = (x1, y1, width, height)
ok = tracker.init(frame, bbox)

### Step 3
#Run the tracker on the video and the selected object. Run the tracker for around 10-15 frames.
cap.set(cv.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
      print("Can't receive frame. Exiting ...")
      break

    ok, bbox = tracker.update(frame)
    print(ok, bbox)

    x1, y1 = bbox[0], bbox[1]
    width, height = bbox[2], bbox[3]

    cv.rectangle(frame, (x1, y1), (x1+width, y1+height), (0, 255, 0), 2)
    cv.imshow("Frame", frame)
    cv.waitKey(3)


cap.release()


"""
  ### Step 6
  Compare the results:
  * Do you see any differences? If so, what are they?
  * Does one tracker perform better than the other? In what way?
  Я использовал два видео для тестирования. В видее traffic_2.mp4 трекеры KCF и MIl 
  работают приблизительно одинаково и не учитывают изменение масштаба приближающегося автомобиля.
  Трекеры CSRT напротив не плохо справляется с трекенгом приближающегося автомобиля и адаптируется 
  к изменению масштаба приближающегося автомобиля.
  В видео traffic_3.mp4 имеется окклюзия в виде разворачивающегося автомобиля, который перекрывает
  трекаемый объект и в результате все трекеры теряют отслеживаемый объект. 
"""



