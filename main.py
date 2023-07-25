import os
import torch
import cv2
import math
import numpy as np
from ssl import _create_unverified_context
from time import time
#from trackers.multi_tracker_zoo import create_tracker
from collections import defaultdict, deque


from xalil_tracker_v1 import EuclideanDistTracker



class SpeedTracker:
    def __init__(self, bot_token, chat_id, source_video_path, video_saving_path, model_path, polygon_points, writer,speed_limit, tracker,max_frame=13):
        _create_default_https_context = _create_unverified_context
        self.writer = writer

        self.tracker =  tracker

        self.source_video_path = source_video_path
        self.video_saving_path = video_saving_path
        self.model_path = model_path
        self.polygon_points = np.array(polygon_points)
        self.max_frame = max_frame
        self.speed_limit = speed_limit

        self.cars = defaultdict(lambda: {"positions": deque(maxlen=self.max_frame), "times": deque(maxlen=self.max_frame)})
        
        self.video_cap = cv2.VideoCapture(self.source_video_path)
        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = video_cap.get(cv2.CAP_PROP_FPS)
        width, height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if writer:
            self.result = cv2.VideoWriter(video_saving_path, cv2.VideoWriter_fourcc(*'mp4v'), 16, (width, height))
        
        #CAR DETECTION MODEL 
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=self.model_path, force_reload=False,device="mps")
        self.names = self.model.names
        self.model.conf = 0.8
        # TRACKER

    def reconnect_video(self, video_cap):
        video_cap.release()
        video_cap = cv2.VideoCapture(self.source_video_path)
        return video_cap

    def process(self):
        count = 0

        prev_time = time()  # Add this line to initialize prev_time
        while self.video_cap.isOpened():
            # try:
            ret, frame = self.video_cap.read()  # Update variable name to self.video_cap
            #     if not ret:
            #         raise Exception("Error reading frame")
            count += 1
            if count % 2 != 0:
                continue

            curr_time = time()
            elapsed_time = curr_time - prev_time
            prev_time = curr_time

            fps = 1.0 / elapsed_time
            cv2.putText(frame, f"FPS: {int(fps)}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 229, 204), 3)

            results = self.model(frame)
            det = results.xyxy[0]

            cv2.polylines(frame, np.int32([self.polygon_points]), True, (55, 155, 255), 3)
            if det is not None and len(det):
                output_cpu = det.cpu().numpy()
                output_list = [[int(x) for x in row[:4]] for row in output_cpu]
                tracked_points = self.tracker.update(output_list)
                for j, (output) in enumerate(tracked_points):
                    x1, y1, x2, y2, obj_id = output  # Unpack the individual elements from objects_bbs_ids

                    cv2.rectangle(frame, (x1, y1), (x2, y2),  (50, 255, 50), 2)
                    cv2.putText(frame, f"car {obj_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,0.7, (255,0,255), 2)

            if not self.writer:
                cv2.imshow("ROI", frame)

            if self.writer:
                print(f"frame {count} writing")
                cv2.imshow("ROI", frame)
                self.result.write(frame)

            if cv2.waitKey(12) == ord('q'):
                break
            # except Exception as e:
            #     print(f"Error: {str(e)}")
            #     print("Reconnecting to video source...")
            #     self.video_cap = self.reconnect_video(self.video_cap)  # Update variable name to self.video_cap

if __name__ == "__main__":
    speed_limit = 50
    writer = False

    bot_token = ""
    chat_id = "-"
    source_video_path="dataset.mp4"

    video_saving_path = "try_except.mp4"
    model_path = "speed_v1.pt"
    xalil_tracker = EuclideanDistTracker()
    polygon_points = [[714, 128], [714, 128], [1562, 127], [1562, 127], [1599, 196], [1599, 196], [692, 197],[692, 197]]
    tracker = SpeedTracker(bot_token, chat_id, source_video_path, video_saving_path, model_path, polygon_points,writer,speed_limit,xalil_tracker)
    tracker.process()
