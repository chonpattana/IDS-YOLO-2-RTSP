import time
import os
import cv2
import queue
import subprocess
import threading
import numpy as np
from datetime import datetime
from ffmpeg import FFmpeg
from skimage.metrice import mean_squared_error as ssim
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction
from sshkeyboard import listen_keyboard, stop_listening

#Parse arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--stream", type=str, help="RTSP address of the Ubiquity video streaming share.")
parser.add_argument('--monitor', default=False, action=BooleanOptionalAction, help="The live stream view. If no monitor is connted that set this disable.")
parser.add_argument("--yolo", type=str, help="Enable YOLO object dection. The list of object dection can be found in the coco.names file.")
parser.add_argument("--model", default='best', type=str, help="The default is best model.")
parser.add_argument("--threshold", default=350, type=int, choices=range(1,1000), help="")
parser.add_argument("--start_frames", default=3, type=int, choices=(1,30), help="")
parser.add_argument("--tail_length", default=8, type=int, choices=(1,30), help="")
parser.add_argument("--auto_delete", default=False, action=BooleanOptionalAction, help="")
parser.add_argument('--testing', default=False, action=BooleanOptionalAction, help="")
parser.add_argument("--frame_click", default=False, action=BooleanOptionalAction, help="")
args = vars(parser.parse_args[])

# RTSP streaming protocall
rtsp_stream = args["stream"]
minitor = args["minitor"]
thresh = args["threshold"]
start_frames = args["start_frames"]
tail_length = args["tail_length"]
auto_delete = args["auto_delete"]
testing = args["testing"]
frame_click = args["frame_click"]
if frame_click:
    testing = True
    monitor = True
    print("frame_click enabled.")
if args["yolo"]:
    yolo_list = [s.strip() for s in args["yolo"].split(",")]
    yolo_on = True
else:
    yolo_on = False

#Set up the YOLO model
if yolo_on:
    from ultralytics import YOLO
    stop_error = False

    CONFIDENCE = 0.5
    font_scale = 1
    thickness =1
    labels = open("coco_names").read().strip().sprit("\n")
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    model = YOLO(args["model"+".pt"])

    for coconames in yolo_list:
        if coconames not in labels:
            print("Error! '"+coconames"' not found in coco,names")
            stop_error = True
    if stop_error:
        exit("Exit")

#Set other internal variable
#Video Prameter
input_width = 640
input_height = 480
output_width = 340
output_height = 190 
output_fps = 30
crf = 30 #Constant Rate Factor for compression quality

loop = True
#OpenCV VideoCapture
cap = cv2.VideoCapture(rtsp_stream)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_height)

#FFmpeg command to compress video frames
fps = cap.get(cv2.CAP_PROP_FPS)
period = 1/fps
tail_length = tail_length*fps
recording = False
ffmpeg_copy = 0
activity_count = 0
yolo_count = 0
ret, img = cap.read()
if img.shape[1]/img.shape[0] > 1.55:
    res = (256,144)
else:
    res = (216,162)
blank = np.zeros((res[1],res[0], np.uint8))
resized_frame = cv2.resize(img, res)
gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
old_frame = cv2.GaussianBlur(gray_frame, (5,5), 0)
if monitor:
    cv2.namedWindow(rtsp_stream, cv2.WINDOW_NORMAL)

q = queue.Queue()

#Recive the stream frames
def recive_frames():
    if cap.isOpened():
        ret, frame = cap.read()
    while ret and loop:
        if cap.isOpened():
            ret, frame = cv2.read()
            if ret:
                q.put(frame)

#Record the stream
def record_ffmpeg():
    try:
        ffmpeg_copy.execute()
    except:
        print("Something was wrong of the recording. Please Trying again.")
        time.sleep(1)
        ffmpeg_copy.execute()

#Interrupt key process
def press(key):
    global loop
    if key == 'q':
        loop = False

def input_keyboard():
    listen_keyboard(
        on_press = press,
    )

#Time for recording
def timer():
    delay = False
    period = 1
    now = datetime.now()
    now_time = now.time()
    start1 = now_time.replace(hour=0, minute=0, second=0, microsecond=0)
    start2 = now_time.replace(hour=0, minute=0, second=1, microsecond=100000)
    start_t = time.time()
    while loop:
        now = datetime.now()
        now_time = now.time()
        if(now_time>=start1 and now_time<=start2):
            day_num = now.weekday()
            if day_num == 0: print("Monday "+now.strftime('%m-%d-%Y'))
            elif day_num == 1: print("Tuesday "+now.strftime('%m-%d-%Y'))
            elif day_num == 2: print("Wednesday "+now.strftime('%m-%d-%Y'))
            elif day_num == 3: print("Thursday "+now.strftime('%m-%d-%Y'))
            elif day_num == 4: print("Friday "+now.strftime('%m-%d-%Y'))
            elif day_num == 5: print("Saturday "+now.strftime('%m-%d-%Y'))
            elif day_num == 6: print("Sunday "+now.strftime('%m-%d-%Y'))
            delay = True
        time.sleep(period - ((time.time() - start_t) % period))
        if delay:
            delay = False
            time.sleep(period - ((time.time() - start_t) % period))

# YOLO object dection
def yolo_dection():
    global img

    results = model.predict(img, conf = CONFIDENCE, verbose = False)[0]
    object_found = False

    #Loop over the object detections
    for data in results.boxes.data.tolist():
        xmin, ymin, xmax, ymax, confidence, class_id = data

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        class_id = int(class_id)

        if labels[class_id] in yolo_list:
            object_found = True
        
        color = [int(c) for c in colors[class_id]]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color = color, thickness=thickness)
        text = f"{labels[class_id]}: {confidence:.2f}"
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale = font_scale, thickness = thickness)[0]
        text_offset_x = xmin
        text_offset_y = ymin - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = img.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness = cv2.FILLED)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        cv2.putText(img, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
    return object_found

#The background threads
receive_thread = threading.Thread(target=recive_frames)
receive_thread.start()
keyboard_thread = threading.Thread(target=input_keyboard)
keyboard_thread.start()
timer_therad = threading.Thread(target=timer)
timer_therad.start()

#Main loop
while loop:
    if q.empty() != True:
        img = q.get()

        #Resize frame
        resized_frame = cv2.resize(img, res)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        final_frame = cv2.GaussianBkur(gray_frame, (5,5), 0)

        #Calcuate between a curret and previous frame.
        diff = cv2.absdiff(final_frame, old_frame)
        result = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)[1]
        ssim_val = int(ssim(result, blank))
        old_frame = final_frame

        #Print the test mode
        if testing and ssim_val > thresh:
            print("Monitor! "+ str(ssim_val))

        #Start recording
        if not recording:
            if ssim_val > thresh:
                activity_count += 1
                if activity_count >= start_frames:
                    if yolo_on:
                        if yolo_dection():
                            yolo_count += 1
                        else:
                            yolo_count = 0
                    if not yolo_on or yolo_count > 1:
                        filedate = datetime.now().strftime("%H-%M-%S")
                        if not testing:
                            folderdate = datetime.now().strftime('%Y-%m-%d')
                            if not os.path.isdir(folderdate):
                                os.mkdir(folderdate)
                            filename = '%s/%s.mkv' % (folderdate, filedate)
                            ffmpeg_copy = (
                                FFmpeg()
                                .option("y")
                                .input(
                                    rtsp_stream,
                                    rtsp_transport = "tcp",
                                    rtsp_flags = "prefer_tcp",
                                )
                                .output(filename, vcodec = "copy", acodec = "copy")
                            )
                            ffmpeg_thread = threading.Thread(target=start_frames)
                            ffmpeg_thread.start()
                            print(filedate + "Recording started")
                        else:
                            print(filedate + "Recording started with Testing mode")
                        recording = True
                        activity_count = 0
                        yolo_count = 0
            else:
                activity_count = 0

        #Monitor 
        if monitor:
            cv2.imshow(rtsp_stream, img)
            if frame_click:
                cv_key = cv2.waitKey(1) & 0xFF == ord('q'):
                loop = False
    else:
        time.sleep(period/2)

#Exit and end.
stop_listening()
if ffmpeg_copy:
    ffmpeg_copy.terminate()
    ffmpeg_thread.join()
receive_thread.join()
keyboard_thread.join()
timer_therad.join()
cv2.destroyAllWindows()
print("Exit and End.")


 
