from operator import index
import os
import pprint
from tkinter.tix import Tree
from turtle import pos, position
import cv2
import time
import argparse
import numpy as np
from hough import *
from scipy.spatial import distance


def find_closest_dot(point, points):
    closest_index = distance.cdist([point], points).argmin()
    return points[closest_index]


def find_dot_position(point, positions):
    points = []
    for position in positions:
        if(position[0] == point):
            return position


def get_valid_points(positions):

    points = []
    for position in positions:
        points.append(position[0])
    return points


def getKey(item):
    return item[0]


def filter_points(points, N):
    subList = [points[n:n+N] for n in range(0, len(points), N)]
    del subList[0]
    for lista in subList:
        lista.sort(key=lambda x: x[0])
        del lista[8]
    return subList


def chess_position_to_point(points):
    points = filter_points(points, 9)

    positions = []
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    numbers = [1, 2, 3, 4, 5, 6, 7, 8]
    counter = 0
    for idx, letter in enumerate(letters):
        for number in numbers:
            positions.append(
                (points[idx][counter % 8], letter + str(number)))
            counter = counter + 1
    return positions


def do_hough(path):
    img, blur = read_image(path)
    edges = canny_edge(blur)
    lines = hough_line(edges)
    h_lines, v_lines = h_v_lines(lines)
    points = line_intersections(h_lines, v_lines)
    points = cluster_points(points)
    return points


# import the necessary packages
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
LABELS = open(labelsPath).read().strip().split("\n")
print(LABELS)
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")


# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov4-custom_best.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov4-custom.cfg"])
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]
# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                             swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()
# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))


# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []
boxes_bottom_left = []
# loop over each of the layer outputs
for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > args["confidence"]:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            # update our list of bounding box coordinates, confidences,
            # and class IDs
            boxes_bottom_left.append(
                (int(centerX - (width / 2)), int(centerY + (height / 2))))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# print(boxes)
# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                        args["threshold"])


# do hough interesections

points = do_hough(args["image"])


positions = chess_position_to_point(points)
points = get_valid_points(positions)
for point in points:
    # print(point)
    cv2.circle(image, (int(point[0]), int(point[1])),
               radius=5, color=(255, 0, 0), thickness=-1)

if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
        cv2.circle(image, boxes_bottom_left[i],
                   radius=7, color=(0, 0, 255), thickness=-1)
        point = find_closest_dot(boxes_bottom_left[i], points)
        print(LABELS[classIDs[i]], confidences[i],
              find_dot_position(point, positions))


cv2.imshow("Image", image)
cv2.waitKey(0)
