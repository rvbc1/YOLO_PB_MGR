#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import argparse
import numpy as np
import json

ap = argparse.ArgumentParser()
ap.add_argument('-in', '--in_video', required=True,
                help = 'path to input image')
ap.add_argument('-out', '--out_video', required=True,
                help = 'path to input image')
ap.add_argument('-json', '--out_json', required=True,
                help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


frames = []

vidcap = cv2.VideoCapture(args.in_video)
success,image = vidcap.read()

#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#out = cv2.VideoWriter('box1.mp4',fourcc, 30, (image.shape[1], image.shape[0]))

outVideo = cv2.VideoWriter(args.out_video,cv2.VideoWriter_fourcc(*'MJPG'), 30, (image.shape[1], image.shape[0]))

count = 0
while success:
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(args.weights, args.config)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)


    obiekt_slownik = {}
    detected = []




    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        obiekt_slownik = {"type": str(classes[class_ids[i]]), "top": int(y),
                    "left": int(x), "bottom": int(y+h), "right": int(x+w) , "right": int(x+w), "confidences": confidences[i]}
        detected.append(obiekt_slownik)
    klatka_slownik = {"frame": count, "detected": detected}
    frames.append(klatka_slownik)

    print("frame ", count);
    #cv2.imwrite("object-detection.jpg", image)
    #cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file  
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (10, 30) 
    fontScale = 1
    color = (0, 0, 255) 
    thickness = 2  
    image = cv2.putText(image, "YOLO, frame %d" % count, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
    outVideo.write(image)  
    success,image = vidcap.read()
    count += 1
outVideo.release()

frames_slownik = {"frames": frames}
with open(args.out_json, 'w') as json_file:
    json.dump(frames_slownik, json_file, indent = 4)

