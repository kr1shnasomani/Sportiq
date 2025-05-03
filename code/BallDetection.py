import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw

from BallTrackNet import BallTrackerNet

def combine_three_frames(frame1, frame2, frame3, width, height):
    img = cv2.resize(frame1, (width, height))
    img = img.astype(np.float32)
    img1 = cv2.resize(frame2, (width, height))
    img1 = img1.astype(np.float32)
    img2 = cv2.resize(frame3, (width, height))
    img2 = img2.astype(np.float32)
    imgs = np.concatenate((img, img1, img2), axis=2)
    imgs = np.rollaxis(imgs, 2, 0)
    return np.array(imgs)

class BallDetector:
    def __init__(self, save_state, out_channels=2):
        self.device = torch.device("cpu")
        self.detector = BallTrackerNet(out_channels=out_channels)

        try:
            saved_state_dict = torch.load(save_state, map_location=torch.device("cpu"), weights_only=False)
        except TypeError:
            saved_state_dict = torch.load(save_state, map_location=torch.device("cpu"))

        self.detector.load_state_dict(saved_state_dict['model_state'])
        self.detector.eval().to(self.device)

        self.current_frame = None
        self.last_frame = None
        self.before_last_frame = None

        self.video_width = None
        self.video_height = None
        self.model_input_width = 640
        self.model_input_height = 360

        self.threshold_dist = 100
        self.xy_coordinates = np.array([[None, None], [None, None]])

        self.bounces_indices = []

    def detect_ball(self, frame):
        if self.video_width is None:
            self.video_width = frame.shape[1]
            self.video_height = frame.shape[0]
        self.last_frame = self.before_last_frame
        self.before_last_frame = self.current_frame
        self.current_frame = frame.copy()

        if self.last_frame is not None:
            frames = combine_three_frames(self.current_frame, self.before_last_frame, self.last_frame,
                                          self.model_input_width, self.model_input_height)
            frames = (torch.from_numpy(frames) / 255).to(self.device)
            x, y = self.detector.inference(frames)
            if x is not None:
                x = int(x * (self.video_width / self.model_input_width))
                y = int(y * (self.video_height / self.model_input_height))

                if self.xy_coordinates[-1][0] is not None:
                    if np.linalg.norm(np.array([x, y]) - self.xy_coordinates[-1]) > self.threshold_dist:
                        x, y = None, None
            self.xy_coordinates = np.append(self.xy_coordinates, np.array([[x, y]]), axis=0)