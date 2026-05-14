# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 13:18:55 2021

@author: droes
"""

import numpy as np
import cv2 # conda install opencv
from matplotlib import pyplot as plt # conda install matplotlib


# For students
def initialize_hist_figure():
    '''
    Usually called only once to initialize the hist figure.
    Do not change the essentials of this function to keep the performance advantages.
    https://www.youtube.com/watch?v=_NNYI8VbFyY
    '''
    fig = plt.figure(figsize=(4.2, 2.4), dpi=100, facecolor='black')
    ax  = fig.add_subplot(111, facecolor='black')
    ax.set_xlim([-0.5, 255.5])
    # fixed size (you can normalize your values between 0, 3 or other ranges to never exceed this limit)
    ax.set_ylim([0,3])
    ax.set_title('RGB Histogram', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('white')
    fig.tight_layout(pad=0.7)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)
    def_x_line = np.arange(0, 256, 1)
    # def_y_line = np.zeros(shape=(256,))
    r_plot = ax.plot(def_x_line, np.zeros(256), 'r', animated=True, linewidth=1.3)[0]
    g_plot = ax.plot(def_x_line, np.zeros(256), 'g', animated=True, linewidth=1.3)[0]
    b_plot = ax.plot(def_x_line, np.zeros(256), 'b', animated=True, linewidth=1.3)[0]
    
    return fig, ax, background, r_plot, g_plot, b_plot



def update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars):
    '''
    Uses the initialized figure to update it accordingly to the new values.
    Do not change the essentials of this function to keep the performance advantages.
    '''
    fig.canvas.restore_region(background)        
    r_plot.set_ydata(r_bars)        
    g_plot.set_ydata(g_bars)        
    b_plot.set_ydata(b_bars)

    ax.draw_artist(r_plot)
    ax.draw_artist(g_plot)
    ax.draw_artist(b_plot)
    fig.canvas.blit(ax.bbox)
    
    

def plot_overlay_to_image(
        np_img,
        plt_figure,
        size=None,
        position='bottom_left',
        margin=20,
        ignore_white=True,
        alpha=1.0):
    '''
    Use this function to create an image overlay.
    You must use a matplotlib figure object.
    Please consider to keep the figure object always outside code loops (performance hint).
    Use this function for example to plot the histogram on top of your image.
    White pixels are ignored (transparency effect)-
    Do not change the essentials of this function to keep the performance advantages.
    '''
    
    rgba_buf = np.asarray(plt_figure.canvas.buffer_rgba())
    imga = rgba_buf[:, :, :3].copy()

    (h_img, w_img, _) = np_img.shape
    if size is None:
        x_pos = 0
        y_pos = 0
        target_width = w_img
        target_height = h_img
    else:
        target_width, target_height = size
        target_width = min(target_width, w_img - 2 * margin)
        target_height = min(target_height, h_img - 2 * margin)

        if position == 'top_left':
            x_pos = margin
            y_pos = margin
        elif position == 'top_right':
            x_pos = w_img - target_width - margin
            y_pos = margin
        elif position == 'bottom_right':
            x_pos = w_img - target_width - margin
            y_pos = h_img - target_height - margin
        else:
            x_pos = margin
            y_pos = h_img - target_height - margin

    if imga.shape[:2] != (target_height, target_width):
        imga = cv2.resize(imga, (target_width, target_height))

    image_region = np_img[y_pos:y_pos + target_height, x_pos:x_pos + target_width]
    overlay_region = imga
    if alpha < 1.0:
        overlay_region = cv2.addWeighted(image_region, 1.0 - alpha, imga, alpha, 0)
    
    if ignore_white:
        # ignore white pixels
        plt_indices = np.argwhere(imga < 255)

        # add only non-white values
        height_indices = plt_indices[:,0]
        width_indices = plt_indices[:,1]
        
        image_region[height_indices, width_indices] = overlay_region[height_indices, width_indices]
    else:
        image_region[:] = overlay_region

    return np_img


def plot_statistics_to_image(
        np_img,
        statistics,
        text_color=(255, 255, 255),
        background_color=(0, 0, 0)):
    '''
    Draws brightness statistics across the top of the image.
    '''
    mean, mode, std_dev, max_value, min_value, entropy = statistics
    stat_text = (
        f'Mean: {mean:.1f}   '
        f'Mode: {mode:.0f}   '
        f'Standard deviation: {std_dev:.1f}   '
        f'Max: {max_value:.0f}   '
        f'Min: {min_value:.0f}   '
        f'Entropy: {entropy:.2f}'
    )

    (h, w, _) = np_img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.68
    thickness = 2
    padding = 12

    text_width, text_height = cv2.getTextSize(stat_text, font, font_scale, thickness)[0]
    while text_width > w - 2 * padding and font_scale > 0.4:
        font_scale -= 0.05
        text_width, text_height = cv2.getTextSize(stat_text, font, font_scale, thickness)[0]

    bar_height = min(h, text_height + 2 * padding)
    bar_region = np_img[:bar_height, :]
    bar_background = np.full_like(bar_region, background_color)
    cv2.addWeighted(bar_background, 0.78, bar_region, 0.22, 0, dst=bar_region)

    y_pos = padding + text_height
    cv2.putText(
        np_img,
        stat_text,
        (padding, y_pos),
        font,
        font_scale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA
    )

    return np_img



def plot_strings_to_image(
        np_img,
        list_of_string,
        text_color=(255, 255, 255),
        right_space=260,
        top_space=50,
        background_color=(0, 0, 0)):
    '''
    Plots the string parameters below each other, starting from top right.
    Use this function for example to plot the default image characteristics.
    Do not change the essentials of this function to keep the performance advantages.
    '''
    y_start = top_space
    min_size = right_space
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    padding = 8
    line_height = 36
    (h, w, c) = np_img.shape
    if w < min_size:
        raise Exception('Image too small in width to print additional text.')
        
    if h < top_space + line_height:
        raise Exception('Image too small in height to print additional text.')
    
    y_pos = y_start
    x_pos = w - min_size

    for text in list_of_string:
        if y_pos >= h:
            break
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size
        top_left = (max(0, x_pos - padding), max(0, y_pos - text_height - padding))
        bottom_right = (
            min(w, x_pos + text_width + padding),
            min(h, y_pos + baseline + padding)
        )

        cv2.rectangle(np_img, top_left, bottom_right, background_color, cv2.FILLED)
        cv2.putText(
            np_img,
            text,
            (x_pos, y_pos),
            font,
            font_scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA
        )
        y_pos += line_height

    return np_img
