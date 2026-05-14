# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
# Note: keyboard library removed due to macOS compatibility issues
# Use Command+C to quit instead

from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, plot_statistics_to_image, plot_strings_to_image, update_histogram
from basics import (
    edge_detection_filter,
    histogram_equalization_filter,
    histogram_figure_numba,
    image_statistics_numba,
    linear_transformation_filter,
    NeuralFaceSmileFilter,
    no_filter,
)


RIGHT_ARROW_KEYS = {3, 83, 63235, 65363, 2555904}


class FilterModeController:
    def __init__(self):
        self.face_smile_filter = NeuralFaceSmileFilter()
        self.filter_modes = [
            ('Original', no_filter),
            ('Face Detection', self.face_smile_filter),
            ('Linear Transformation', linear_transformation_filter),
            ('Histogram Equalization', histogram_equalization_filter),
            ('Edge Detection', edge_detection_filter),
        ]
        self.current_index = 0

    @property
    def current_name(self):
        return self.filter_modes[self.current_index][0]

    def apply_current_filter(self, sequence):
        filter_func = self.filter_modes[self.current_index][1]
        return filter_func(sequence)

    def handle_key_press(self, key):
        if key in RIGHT_ARROW_KEYS:
            self.current_index = (self.current_index + 1) % len(self.filter_modes)
            print(f'Filter mode: {self.current_name}')


# Example function
# You can use this function to process the images from opencv
# This function must be implemented as a generator function
def custom_processing(img_source_generator, filter_controller):
    # use this figure to plot your histogram
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()
    
    for sequence in img_source_generator:
        # Call your custom processing methods here! (e. g. filters)
        sequence = filter_controller.apply_current_filter(sequence)
        

        # Example of keyboard is pressed
        # Note: keyboard library removed due to macOS compatibility issues
        # Use alternative input methods as needed
        # if keyboard.is_pressed('h') :
        #     print('h pressed')
            

        ###
        ### Histogram overlay example (with live RGB data)
        ###
        
        # Load the histogram and brightness statistics
        r_bars, g_bars, b_bars = histogram_figure_numba(sequence)
        image_statistics = image_statistics_numba(sequence)
        
        # Update the histogram with new data
        update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)
        
        # uses the figure to create the histogram overlay in the corner
        sequence = plot_overlay_to_image(
            sequence,
            fig,
            size=(380, 220),
            position='bottom_left',
            margin=20,
            ignore_white=False,
            alpha=0.9
        )

        # print brightness statistics on top of the screen
        sequence = plot_statistics_to_image(sequence, image_statistics)

        sequence = plot_strings_to_image(
            sequence,
            [f'Filter: {filter_controller.current_name}'],
            right_space=420,
            top_space=70
        )
        
        ###
        ### END Histogram overlay example
        ###

        
        # Display text example
        #display_text_arr = ["Test", "abc"]
        #sequence = plot_strings_to_image(sequence, display_text_arr)

        
        # Make sure to yield your processed image
        yield sequence



def main():
    # change according to your settings
    width = 1280
    height = 720
    fps = 30
    
    # Define your virtual camera
    vc = VirtualCamera(fps, width, height)
    filter_controller = FilterModeController()
    print(f'Filter mode: {filter_controller.current_name}')
    print('Press the right arrow in the preview window to switch filter modes.')
    
    vc.virtual_cam_interaction(
        custom_processing(
            # either camera stream
            vc.capture_cv_video(camera_id=None, bgr_to_rgb=True),
            filter_controller
            
            # or your window screen
            # vc.capture_screen()
        ),
        preview=True,
        on_key_press=filter_controller.handle_key_press
    )

if __name__ == "__main__":
    main()
