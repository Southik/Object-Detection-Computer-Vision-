# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:58:41 2021

@author: droes
"""

import pyvirtualcam
import numpy as np
import cv2 # conda install opencv
from PIL import ImageGrab # conda install pillow
from matplotlib import pyplot as plt # conda install matplotlib
import time
import sys
# keyboard library removed due to macOS compatibility issues


class VirtualCamera:
    def __init__(self, fps, width, height):
        self.fps = fps
        self.width = width
        self.height = height

    def _prepare_frame(self, frame):
        '''
        pyvirtualcam requires exactly (height, width, 3) uint8 frames.
        Some cameras ignore requested OpenCV dimensions, so normalize here.
        '''
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim != 3:
            raise ValueError(f'Unsupported frame dimensions: {frame.shape}')

        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        elif frame.shape[2] != 3:
            raise ValueError(f'Unsupported channel count: {frame.shape}')

        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(frame)

    def _open_cv_camera(self, camera_id):
        avfoundation_backend = getattr(cv2, 'CAP_AVFOUNDATION', 0)
        cv_vid = cv2.VideoCapture(camera_id, avfoundation_backend)

        if not cv_vid.isOpened():
            cv_vid.release()
            cv_vid = cv2.VideoCapture(camera_id)

        if not cv_vid.isOpened():
            cv_vid.release()
            return None

        cv_vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cv_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cv_vid.set(cv2.CAP_PROP_FPS, self.fps)

        # AVFoundation on macOS can fail to deliver frames when forced to MJPG.
        if sys.platform != 'darwin':
            cv_vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        return cv_vid

    def _read_camera_frame(self, cv_vid, retries=30, retry_delay=0.1):
        for _ in range(retries):
            ret, frame = cv_vid.read()
            if ret and frame is not None:
                return frame
            time.sleep(retry_delay)
        return None

    def _select_cv_camera(self, camera_id):
        camera_ids = range(6) if camera_id is None else [camera_id]

        for candidate_id in camera_ids:
            cv_vid = self._open_cv_camera(candidate_id)
            if cv_vid is None:
                continue

            frame = self._read_camera_frame(cv_vid)
            if frame is not None:
                width = int(cv_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cv_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps_in = cv_vid.get(cv2.CAP_PROP_FPS)
                print(f'Using input camera {candidate_id}: ({width}x{height} @ {fps_in}fps)')
                return candidate_id, cv_vid, frame

            print(f'Camera {candidate_id} opened but did not return frames; trying next camera.')
            cv_vid.release()

        if camera_id is None:
            raise RuntimeError(
                'No camera returned frames. Close apps that may be using the webcam, '
                'then try again. If needed, set a specific camera index in run.py.'
            )

        raise RuntimeError(
            f'Camera {camera_id} did not return frames. Try camera_id=None for auto-detection, '
            'or change the camera index in run.py.'
        )
        
    def capture_screen(self, plt_inside=False, alt_width=0, alt_height=0):
        '''
        Represents the content of the primary monitor.
        Can be used to quickly test your application.
        '''
        
        width = alt_width if alt_width > 0 else self.width
        height = alt_height if alt_height > 0 else self.height
        while True:
            # grab is a slow method!
            img = ImageGrab.grab(bbox=(0, 0, width, height)) #x, y, w, h
            img_np = np.array(img)
            #img_np = np.zeros(shape=(height, width, 3), dtype=np.uint8)
            if plt_inside:
                plt.imshow(img_np)
                plt.axis('off')
                plt.show()
            yield img_np

            
    def capture_cv_video(self, camera_id=None, bgr_to_rgb=False):
        '''
        Establishes the connection to the camera via opencv
        Source: https://github.com/letmaik/pyvirtualcam/blob/master/samples/webcam_filter.py
        '''
        selected_camera_id, cv_vid, first_frame = self._select_cv_camera(camera_id)
        
        try:
            frame = first_frame
            while True:
                if bgr_to_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                # Note: keyboard quit removed due to macOS compatibility
                # Use Ctrl+C to quit instead
                yield self._prepare_frame(frame)

                frame = self._read_camera_frame(cv_vid)
                if frame is None:
                    raise RuntimeError(
                        f'Input camera {selected_camera_id} stopped returning frames. '
                        'Close any other camera apps and run this script again.'
                    )
        finally:
            cv_vid.release()

    
    def _handle_preview_key(self, key, on_key_press):
        if key == -1:
            return False

        if key & 0xFF == ord('q'):
            return True

        if on_key_press is not None:
            on_key_press(key)

        return False

    def virtual_cam_interaction(self, img_generator, print_fps=True, preview=False, on_key_press=None):
        '''
        Provides a virtual camera.
        img_generator must represent a function that acts as a generator and returns image data.
        '''
        if preview:
            print('Quit camera stream with Ctrl+C, or press q in the preview window')
        else:
            print('Quit camera stream with Ctrl+C')
        img_generator = iter(img_generator)
        first_img = self._prepare_frame(next(img_generator))
        with pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps, print_fps=print_fps) as cam:
            print(f'Using virtual camera: {cam.device} ({cam.backend} backend)')
            cam.send(first_img)
            if preview:
                cv2.imshow('Processed camera preview', cv2.cvtColor(first_img, cv2.COLOR_RGB2BGR))
                key = cv2.waitKeyEx(1)
                if self._handle_preview_key(key, on_key_press):
                    return
            cam.sleep_until_next_frame()
            try:
                for img in img_generator:
                    # provide the image
                    img = self._prepare_frame(img)
                    cam.send(img)
                    if preview:
                        cv2.imshow('Processed camera preview', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        key = cv2.waitKeyEx(1)
                        if self._handle_preview_key(key, on_key_press):
                            break
                    # wait for next frame (fps dependent)
                    cam.sleep_until_next_frame()
            finally:
                if preview:
                    cv2.destroyWindow('Processed camera preview')
