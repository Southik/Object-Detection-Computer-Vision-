# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:18:29 2021

@author: droes
"""
from numba import njit # conda install numba
import numpy as np
import math
import cv2
from pathlib import Path
from urllib.request import urlretrieve

@njit
def histogram_figure_numba(np_img):
    '''
    Jit compiled function to increase performance.
    Use some loops insteads of purely numpy functions.
    If you face some compile errors using @njit, see: https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html
    In case you dont need performance boosts, remove the njit flag above the function
    Do not use cv2 functions together with @njit
    '''
    r_bars = np.zeros(256, dtype=np.float64)
    g_bars = np.zeros(256, dtype=np.float64)
    b_bars = np.zeros(256, dtype=np.float64)

    height = np_img.shape[0]
    width = np_img.shape[1]

    for y in range(height):
        for x in range(width):
            r_bars[np_img[y, x, 0]] += 1.0
            g_bars[np_img[y, x, 1]] += 1.0
            b_bars[np_img[y, x, 2]] += 1.0

    max_count = 1.0
    for i in range(256):
        if r_bars[i] > max_count:
            max_count = r_bars[i]
        if g_bars[i] > max_count:
            max_count = g_bars[i]
        if b_bars[i] > max_count:
            max_count = b_bars[i]

    for i in range(256):
        r_bars[i] = (r_bars[i] / max_count) * 3.0
        g_bars[i] = (g_bars[i] / max_count) * 3.0
        b_bars[i] = (b_bars[i] / max_count) * 3.0

    return r_bars, g_bars, b_bars


@njit
def image_statistics_numba(np_img):
    '''
    Calculates brightness statistics from the current RGB frame.
    The brightness value is an integer luminance estimate from 0 to 255.
    '''
    gray_hist = np.zeros(256, dtype=np.float64)
    height = np_img.shape[0]
    width = np_img.shape[1]
    total_pixels = height * width

    for y in range(height):
        for x in range(width):
            r = int(np_img[y, x, 0])
            g = int(np_img[y, x, 1])
            b = int(np_img[y, x, 2])
            gray = (299 * r + 587 * g + 114 * b) // 1000
            gray_hist[gray] += 1.0

    min_value = 0
    max_value = 0
    mode_value = 0
    mode_count = 0.0
    sum_values = 0.0

    for i in range(256):
        count = gray_hist[i]
        if count > 0.0:
            if gray_hist[min_value] == 0.0:
                min_value = i
            max_value = i
            sum_values += i * count
            if count > mode_count:
                mode_count = count
                mode_value = i

    mean = sum_values / total_pixels

    variance = 0.0
    entropy = 0.0
    log_2 = math.log(2.0)

    for i in range(256):
        count = gray_hist[i]
        if count > 0.0:
            diff = i - mean
            variance += count * diff * diff
            probability = count / total_pixels
            entropy -= probability * (math.log(probability) / log_2)

    std_dev = math.sqrt(variance / total_pixels)

    return mean, mode_value, std_dev, max_value, min_value, entropy


def no_filter(np_img):
    '''
    Leaves the image unchanged.
    '''
    return np_img


def linear_transformation_filter(np_img, alpha=1.25, beta=12):
    '''
    Applies a simple linear brightness/contrast transform:
    output = alpha * input + beta.
    '''
    return cv2.convertScaleAbs(np_img, alpha=alpha, beta=beta)


def histogram_equalization_filter(np_img):
    '''
    Equalizes image brightness while preserving color as much as possible.
    '''
    ycrcb_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)


def edge_detection_filter(np_img, low_threshold=80, high_threshold=160):
    '''
    Applies Canny edge detection and returns a 3-channel RGB edge image.
    '''
    gray_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(gray_img, low_threshold, high_threshold)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


class NeuralFaceSmileFilter:
    '''
    Detects faces with an OpenCV DNN face detector and replaces each face with a smiley.
    The detector uses pretrained ResNet-SSD Caffe weights.
    '''
    PROTOTXT_NAME = 'deploy.prototxt'
    MODEL_NAME = 'res10_300x300_ssd_iter_140000.caffemodel'
    PROTOTXT_URL = (
        'https://raw.githubusercontent.com/opencv/opencv/master/'
        'samples/dnn/face_detector/deploy.prototxt'
    )
    MODEL_URL = (
        'https://raw.githubusercontent.com/opencv/opencv_3rdparty/'
        'dnn_samples_face_detector_20170830/'
        'res10_300x300_ssd_iter_140000.caffemodel'
    )

    def __init__(self, confidence_threshold=0.6):
        self.confidence_threshold = confidence_threshold
        self.prototxt_path = Path(__file__).resolve().parent / 'models' / self.PROTOTXT_NAME
        self.model_path = Path(__file__).resolve().parent / 'models' / self.MODEL_NAME
        self.net = None

    def __call__(self, np_img):
        self._ensure_net()
        h_img, w_img, _ = np_img.shape
        bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        blob = cv2.dnn.blobFromImage(
            bgr_img,
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        output = np_img.copy()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidence_threshold:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w_img, h_img, w_img, h_img])
            x1, y1, x2, y2 = box.astype(np.int32)
            self._draw_smiley(output, x1, y1, x2 - x1, y2 - y1)

        return output

    def _ensure_net(self):
        self._ensure_model_files()
        if self.net is None:
            self.net = cv2.dnn.readNetFromCaffe(str(self.prototxt_path), str(self.model_path))

    def _ensure_model_files(self):
        self._ensure_file(self.prototxt_path, self.PROTOTXT_URL, min_size=1_000)
        self._ensure_file(self.model_path, self.MODEL_URL, min_size=1_000_000)

    def _ensure_file(self, path, url, min_size):
        if path.exists() and path.stat().st_size > min_size:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Downloading neural face detection file to {path} ...')
        try:
            urlretrieve(url, path)
        except Exception as exc:
            raise RuntimeError(
                'Could not download the pretrained neural face detector. '
                f'Download it manually from {url} and save it as {path}. '
                f'Original error: {exc}'
            ) from exc

        if path.stat().st_size <= min_size:
            raise RuntimeError(
                f'The downloaded file at {path} is too small. '
                'It may be a failed download.'
            )

    def _draw_smiley(self, np_img, x, y, width, height):
        h_img, w_img, _ = np_img.shape
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img - 1, x + width)
        y2 = min(h_img - 1, y + height)

        if x2 <= x1 or y2 <= y1:
            return

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = max(8, int(min(x2 - x1, y2 - y1) * 0.58))
        radius = min(radius, center_x, center_y, w_img - 1 - center_x, h_img - 1 - center_y)

        if radius <= 0:
            return

        yellow = (255, 220, 0)
        black = (0, 0, 0)
        rosy = (255, 120, 120)

        cv2.circle(np_img, (center_x, center_y), radius, yellow, cv2.FILLED, lineType=cv2.LINE_AA)
        cv2.circle(np_img, (center_x, center_y), radius, black, max(2, radius // 18), lineType=cv2.LINE_AA)

        eye_radius = max(2, radius // 9)
        eye_y = center_y - radius // 4
        left_eye = (center_x - radius // 3, eye_y)
        right_eye = (center_x + radius // 3, eye_y)
        cv2.circle(np_img, left_eye, eye_radius, black, cv2.FILLED, lineType=cv2.LINE_AA)
        cv2.circle(np_img, right_eye, eye_radius, black, cv2.FILLED, lineType=cv2.LINE_AA)

        cheek_radius = max(2, radius // 11)
        cheek_y = center_y + radius // 8
        cv2.circle(np_img, (center_x - radius // 2, cheek_y), cheek_radius, rosy, cv2.FILLED, lineType=cv2.LINE_AA)
        cv2.circle(np_img, (center_x + radius // 2, cheek_y), cheek_radius, rosy, cv2.FILLED, lineType=cv2.LINE_AA)

        smile_center = (center_x, center_y + radius // 8)
        smile_axes = (max(4, radius // 2), max(3, radius // 3))
        cv2.ellipse(
            np_img,
            smile_center,
            smile_axes,
            0,
            20,
            160,
            black,
            max(2, radius // 14),
            lineType=cv2.LINE_AA
        )


####

### All other basic functions

####
