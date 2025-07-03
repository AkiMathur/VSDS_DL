import streamlit as st
import tempfile
import cv2
import math
import time
import numpy as np
import os
import matplotlib.pyplot as plt

import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv, C2f, Bottleneck, SPPF
from torch.nn.modules.container import Sequential

torch.serialization.add_safe_globals([
    DetectionModel,
    Conv,
    C2f,
    Bottleneck,
    SPPF,
    Sequential
])

from ultralytics import YOLO
import pandas as pd
import utilsss
from PIL import Image
model=YOLO('yolov8s.pt')

 
DEMO_VIDEO = os.path.join('highway.mp4')
DEMO_IMG = os.path.join('50.jpg')

def main():
    st.title('Custom Object Detection using Streamlit')

    st.sidebar.title('Custom Vehicle Detection')
    option = st.sidebar.selectbox('What System Would You Like',('Vehicle Count', 'Number Plate OCR', 'VSDS','VSDS + ANPR'))
    #st.write('You selected:', option)
    #use_webcam = st.sidebar.button('Use Webcam')

    if option == "Vehicle Count":
        st.write('You selected:', option)

        video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ],key=0)

        tfflie = tempfile.NamedTemporaryFile(delete=False)

        if not video_file_buffer:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO
        
        else:
            tfflie.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tfflie.name)

        

        st.sidebar.text('Input Video')
        st.sidebar.video(tfflie.name)

        custom_lines = st.sidebar.checkbox('Use Custom lines')
        if custom_lines:

            red_y = st.sidebar.number_input("Red Line Y-axis:",min_value=0,max_value=500,value=198)
            red_x1 = st.sidebar.number_input("Red Line X1:",min_value=0,max_value=1020,value=172)
            red_x2 = st.sidebar.number_input("Red Line X2:",min_value=0,max_value=1020,value=774)
            blue_y = st.sidebar.number_input("Blue Line Y-axis:",min_value=0,max_value=500,value=268)
            blue_x1 = st.sidebar.number_input("Blue Line X1:",min_value=0,max_value=1020,value=8)
            blue_x2 = st.sidebar.number_input("Blue Line X2:",min_value=0,max_value=1020,value=927)

        class_list = {2:'car',5:'bus', 7:'truck'}
        #class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        custom_classes = st.sidebar.checkbox('Use Custom Classes')
        if custom_classes:

            assigned_class = st.sidebar.multiselect('Select The Custom Classes',list(class_list.values()),default='car')

        st.markdown(' ## Output')
        stop_button = st.sidebar.button('Stop Processing')
        if stop_button:
            st.stop()
        stframe = st.empty()
        save_video = st.button('Save Results')

        #values 
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        codec = cv2.VideoWriter_fourcc('V','P','0','9')
        out = cv2.VideoWriter('output1.webm', codec, fps, (width, height))

        
        tracker=utilsss.Tracker()
        count=utilsss.i_m_speed("count")
        cap=vid
        down = utilsss.i_m_speed("down")
        up = utilsss.i_m_speed("up")
        counter_down = utilsss.i_m_speed("counter_down")
        counter_up = utilsss.i_m_speed("counter_up")

        
        offset = 6

        if custom_lines:
            red_line_y = red_y
            blue_line_y = blue_y
            rx1 = red_x1
            rx2 = red_x2
            bx1 = blue_x1
            bx2 = blue_x2
        else:
            red_line_y = 198
            blue_line_y = 268
            rx1 = 172
            rx2 = 774
            bx1 = 8
            bx2 = 927

        while vid.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            # if count % 2 != 0:
            #     continue
            frame = cv2.resize(frame, (1020, 500))

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
                c = utilsss.class_list[d]
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
                        elapsed_time=time.time() - down[id]  # current time when vehicle touch the second line. Also we a re minusing the previous time ( current time of line 1)
                        if counter_down.count(id)==0:
                            counter_down.append(id)
                            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                            cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                    

                    
                #####going UP blue line#####     
                if blue_line_y<(cy+offset) and blue_line_y > (cy-offset):
                    up[id]=time.time()
                if id in up:

                    if red_line_y<(cy+offset) and red_line_y > (cy-offset):
                        elapsed1_time=time.time() - up[id]
                        # formula of speed= distance/time 
                        if counter_up.count(id)==0:
                            counter_up.append(id)      
                            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                            cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)


        
            text_color = (0, 0, 0)  # Black color for text
            yellow_color = (0, 255, 255)  # Yellow color for background
            red_color = (0, 0, 255)  # Red color for lines
            blue_color = (255, 0, 0)  # Blue color for lines

            cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)

            cv2.line(frame, (rx1, red_line_y), (rx2, red_line_y), red_color, 2)
            cv2.putText(frame, ('Red Line'), (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            cv2.line(frame, (bx1, blue_line_y), (bx2, blue_line_y), blue_color, 2)
            cv2.putText(frame, ('Blue Line'), (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            cv2.putText(frame, ('Going Down - ' + str(len(counter_down))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            cv2.putText(frame, ('Going Up - ' + str(len(counter_up))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)


            #out.write(frame)

            #cv2.imshow("frames", frame)
            stframe.image(frame,channels = 'BGR',use_column_width=True)


        
    elif option == "Number Plate OCR":
        st.write('You selected:', option)

        img_file_buffer = st.sidebar.file_uploader("Upload an image", type=['png', 'jpg'])

        tfflie = tempfile.NamedTemporaryFile(delete=False)

        if not img_file_buffer:
            img = DEMO_IMG
        
        else:
            file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)  
            st.sidebar.image(opencv_image,channels="BGR") 


        st.sidebar.text('Input Image')
        
        start_b = st.sidebar.button('Start Processing')
        if start_b:
            #image = cv2.imread(image)
            number_plate = utilsss.image_ocr(opencv_image)
            norm_img = np.zeros((number_plate.shape[0], number_plate.shape[1]))
            img = cv2.normalize(number_plate, norm_img, 0, 255, cv2.NORM_MINMAX)
            char = utilsss.segment_characters(img)
            st.write('Plate Number is :', utilsss.show_results(char))



        st.markdown(' ## Output')
        stframe = st.empty()
        save_video = st.button('Save Results')



    elif option == "VSDS":
        st.write('You selected:', option)

        video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ],key=0)

        tfflie = tempfile.NamedTemporaryFile(delete=False)

        if not video_file_buffer:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO
        
        else:
            tfflie.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tfflie.name)

        

        st.sidebar.text('Input Video')
        st.sidebar.video(tfflie.name)

        class_list = {2:'car',5:'bus', 7:'truck'}
        #class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        custom_classes = st.sidebar.checkbox('Use Custom Classes')
        if custom_classes:

            assigned_class = st.sidebar.multiselect('Select The Custom Classes',list(class_list.values()),default='car')

        st.markdown(' ## Output')
        stop_button = st.sidebar.button('Stop Processing')
        if stop_button:
            st.stop()
        stframe = st.empty()
        save_video = st.button('Save Results')

        #values 
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        codec = cv2.VideoWriter_fourcc('V','P','0','9')
        out = cv2.VideoWriter('output1.webm', codec, fps, (width, height))

        
        tracker=utilsss.Tracker()
        count=utilsss.i_m_speed("count")
        cap=vid
        down = utilsss.i_m_speed("down")
        up = utilsss.i_m_speed("up")
        counter_down = utilsss.i_m_speed("counter_down")
        counter_up = utilsss.i_m_speed("counter_up")

        red_line_y = 198
        blue_line_y = 268
        offset = 6

        while vid.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            # if count % 2 != 0:
            #     continue
            frame = cv2.resize(frame, (1020, 500))

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
                c = utilsss.class_list[d]
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

            cv2.line(frame, (172, 198), (774, 198), red_color, 2)
            cv2.putText(frame, ('Red Line'), (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            cv2.line(frame, (8, 268), (927, 268), blue_color, 2)
            cv2.putText(frame, ('Blue Line'), (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            cv2.putText(frame, ('Going Down - ' + str(len(counter_down))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            cv2.putText(frame, ('Going Up - ' + str(len(counter_up))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)


            #out.write(frame)

            #cv2.imshow("frames", frame)
            stframe.image(frame,channels = 'BGR',use_column_width=True)
    else:
        st.write('You selected:', option)
        
        video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ],key=0)

        tfflie = tempfile.NamedTemporaryFile(delete=False)

        if not video_file_buffer:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO
        
        else:
            tfflie.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tfflie.name)

        

        st.sidebar.text('Input Video')
        st.sidebar.video(tfflie.name)

        class_list = {2:'car',5:'bus', 7:'truck'}
        #class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        custom_classes = st.sidebar.checkbox('Use Custom Classes')
        if custom_classes:

            assigned_class = st.sidebar.multiselect('Select The Custom Classes',list(class_list.values()),default='car')

        st.markdown(' ## Output')
        stop_button = st.sidebar.button('Stop Processing')
        if stop_button:
            st.stop()
        stframe = st.empty()
        save_video = st.button('Save Results')

        #values 
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        codec = cv2.VideoWriter_fourcc('V','P','0','9')
        out = cv2.VideoWriter('output1.webm', codec, fps, (width, height))

        
        tracker=utilsss.Tracker()
        count=utilsss.i_m_speed("count")
        cap=vid
        down = utilsss.i_m_speed("down")
        up = utilsss.i_m_speed("up")
        counter_down = utilsss.i_m_speed("counter_down")
        counter_up = utilsss.i_m_speed("counter_up")

        red_line_y = 450
        blue_line_y = 650
        offset = 6

        while vid.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            # if count % 2 != 0:
            #     continue
            cropped = frame[500:2160, 10:2500]
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
                c = utilsss.class_list[d]
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
                        elapsed_time=time.time() - down[id]  # current time when vehicle touch the second line. Also we a re minusing the previous time ( current time of line 1)
                        if counter_down.count(id)==0:
                            counter_down.append(id)
                            distance = 10 # meters 
                            a_speed_ms = distance / elapsed_time
                            a_speed_kh = a_speed_ms * 36  # this will give kilometers per hour for each vehicle. This is the condition for going downside
                            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                            crop_img = frame[y3:y4, x3:x4]
                            image = (crop_img)
                            number_plate = utilsss.image_ocr(image)
                            if number_plate is not None:
                                norm_img = np.zeros((number_plate.shape[0], number_plate.shape[1]))
                                img = cv2.normalize(number_plate, norm_img, 0, 255, cv2.NORM_MINMAX)
                                char = utilsss.segment_characters(img)
                                plate = utilsss.show_results(char)
                            else:
                                plate = "Number plate not readable"
                            ##crop_img = cv2.resize(crop_img, (800, 800))
                            #frame_filename = f'detected_frames/{count}.jpg'
                            #cv2.imwrite(frame_filename, crop_img)
                            cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                            cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                            cv2.putText(frame,plate,(x3+50,y3),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),3)

                    
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


            #out.write(frame)

            #cv2.imshow("frames", frame)
            stframe.image(frame,channels = 'BGR',use_column_width=True)

        
        
    cv2.destroyAllWindows()










main()
