import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv, C2f, Bottleneck, SPPF
from torch.nn.modules.container import Sequential
from ultralytics import YOLO  # Must come after registration

import cv2
import math
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score

# ✅ Wrap model loading in context manager
with torch.serialization.safe_globals([DetectionModel, Conv, C2f, Bottleneck, SPPF, Sequential]):
    model = YOLO('yolov8s.pt')

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
                'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
#                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
    



def i_m_speed(a):
    if a == "count":
        return 0
    if a == "down":
        return {}
    if a == "up":
        return {}
    if a == "counter_down":
        return []
    if a == "counter_up":
        return []
    red_line_y = 450
    blue_line_y = 650

    tracker=Tracker()
    count=0
    

    down = {}
    up = {}
    counter_down = []
    counter_up = []

    red_line_y = 450
    blue_line_y = 650
    offset = 6

    # Create a folder to save frames
    '''if not os.path.exists('detected_frames'):
        os.makedirs('detected_frames')'''

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 500))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        d_count=0
        # if count % 2 != 0:
        #     continue
        #height, width, _ = frame.shape

        # Resize the frame
        '''scale_percent = 35
        new_width = int(width * scale_percent / 100)
        new_height = int(height * scale_percent / 100)
        dim = (new_width, new_height)'''
        #frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        '''centerX,centerY=int(height/2),int(width/2)
        radiusX,radiusY= int(scale*height/100),int(scale*width/100)

        minX,maxX=centerX-radiusX,centerX+radiusX
        minY,maxY=centerY-radiusY,centerY+radiusY'''
        
        #cropped = frame[minX:maxX, minY:maxY]
        cropped = frame[500:2160, 10:2500]
        #resized_cropped = cv2.resize(cropped, (width, height))
        frame = cv2.resize(cropped, (1000, 800))

        results = model.predict(frame)
        a = results[0].boxes.data
        a = a.detach().cpu().numpy()
        px = pd.DataFrame(a).astype("float")
        list = []

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if 'car' in c:
                list.append([x1, y1, x2, y2])
        bbox_id = tracker.update(list)

        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2

            if red_line_y<(cy+offset) and red_line_y > (cy-offset):
                down[id]=time.time()   # current time when vehichle touch the first line
            if id in down:
          
                if blue_line_y<(cy+offset) and blue_line_y > (cy-offset):
                    d_count+=1
                    elapsed_time=time.time() - down[id]  # current time when vehicle touch the second line. Also we a re minusing the previous time ( current time of line 1)
                    if counter_down.count(id)==0:
                        counter_down.append(id)
                        distance = 10 # meters 
                        a_speed_ms = distance / elapsed_time
                        a_speed_kh = a_speed_ms * 36  # this will give kilometers per hour for each vehicle. This is the condition for going downside
                        cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                        #crop_img = frame[y3:y4, x3:x4]
                        ##crop_img = cv2.resize(crop_img, (800, 800))
                        #frame_filename = f'detected_frames/{count}.jpg'
                        #cv2.imwrite(frame_filename, crop_img)
                        cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                        cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                

                
            #####going UP blue line#####     
            if blue_line_y<(cy+offset) and blue_line_y > (cy-offset):
                up[id]=time.time()
            if id in up:

                if red_line_y<(cy+offset) and red_line_y > (cy-offset):
                    elapsed1_time=time.time() - up[id]
                    # formula of speed= distance/time 
                    if counter_up.count(id)==0:
                        counter_up.append(id)      
                        distance1 = 10 # meters  (Distance between the 2 lines is 10 meters )
                        a_speed_ms1 = distance1 / elapsed1_time
                        a_speed_kh1 = a_speed_ms1 * 36
                        cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                        cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                        cv2.putText(frame,str(int(a_speed_kh1))+'Km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)



    
        text_color = (0, 0, 0)  # Black color for text
        yellow_color = (0, 255, 255)  # Yellow color for background
        red_color = (0, 0, 255)  # Red color for lines
        blue_color = (255, 0, 0)  # Blue color for lines

        cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)

        cv2.line(frame, (8, 450), (970, 450), red_color, 2)
        cv2.putText(frame, ('Red Line'), (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        cv2.line(frame, (8, 650), (1000, 650), blue_color, 2)
        cv2.putText(frame, ('Blue Line'), (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        cv2.putText(frame, ('Going Down - ' + str(len(counter_down))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(frame, ('Going Up - ' + str(len(counter_up))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        # Save frame
        '''frame_filename = f'detected_frames/frame_{count}.jpg'
        cv2.imwrite(frame_filename, frame)'''

        out.write(frame)

        cv2.imshow("frames", frame)
        if cv2.waitKey(10) & 0xFF == 27:
        #if cv2.waitKey(0) & 0xFF == 27:
            break
    

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def image_ocr(img):
    harcascade = r"C:\Users\Akshit Mathur\Desktop\Deep Learning\haarcascade_russian_plate_number.xml"
    img = cv2.resize(img, (800, 800))
    

    min_area = 500
    count = 0
    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x,y,w,h) in plates:
        area = w * h

        if area > min_area:
            img_roi = img[y: y+h, x:x+w]
            return img_roi
        
# Match contours to license plate or character template
# Match contours to license plate or character template
def find_contours(dimensions, img) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     
    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[3:15]
    
    #ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            #cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            #plt.imshow(ii, cmap='gray')

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # List that stores the character's binary image (unsorted)
            
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    #plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res

def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    img_binary_lp = cv2.adaptiveThreshold(img_gray_lp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19,2)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    #plt.imshow(img_binary_lp, cmap='gray')
    #plt.show()
    #cv2.imwrite('contour.jpg',img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list


def f1score(y, y_pred):
  return f1_score(y, tf.math.argmax(y_pred, axis=1), average='micro') 

def custom_f1score(y, y_pred):
  return tf.py_function(f1score, (y, y_pred), tf.double)

def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img

def show_results(char):
    nmodel = keras.models.load_model('deep.h5',custom_objects={"custom_f1score": custom_f1score })
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char): #iterating over the characters
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        predicted_probabilities = nmodel.predict(img)  # x_test is your input data
        predicted_classes = np.argmax(predicted_probabilities, axis=-1)
        y_ = predicted_classes[0] #predicting the class
        character = dic[y_] #
        output.append(character) #storing the result in a list

    plate_number = ''.join(output)

    return plate_number

