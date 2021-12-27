from utils.io_classes.base_io import BaseIO

import numpy as np
import cv2
import pyexr
import os


class ImageIO(BaseIO):
    """
    Image Sequence Input & Output

    Arguments
        output_path (str): The path to write the images to
    """
    def __init__(self, output_path):
        super(ImageIO, self).__init__(output_path)

        self.count = 0

    def set_input(self, input_folder):
        """
        Load file paths with correct image extensions and feed data.

        Arguments:
            input_folder (str): The folder to recursively grab paths from.
        """
        images = []
        for root, _, files in os.walk(input_folder):
            for file in sorted(files):
                if file.split('.')[-1].lower() in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tga', 'exr']:
                    images.append(os.path.join(root, file))
                if file.split('.')[-1].lower() in ['exr']:
                    self.in_exr = True
                else: self.in_exr = False
        self.feed_data(images)

    def save_frames(self, frames, exr):
        """
        Save frame data as images.

        Arguments:
            frames (ndarray, list): The image data to be written
        """
        self.exr = exr
        
        if not isinstance(frames, list):
            frames = [frames]
        # TODO: Re-add ability to save with original name
        if self.exr is True:
            for img in frames:
                pyexr.write(os.path.join(self.output_path, 
                                     f'{(self.count):08}.exr'), img,
                                     precision = pyexr.HALF, 
                                     compression = pyexr.PXR24_COMPRESSION)

                self.count += 1
        else:
            for img in frames:
                cv2.imwrite(os.path.join(self.output_path,
                                     f'{(self.count):08}.png'), img)
                self.count += 1
        
        

    def __getitem__(self, idx):
        if self.in_exr is True:
            pytest= pyexr.open(self.data[idx])
            return pytest.get('default')
        else:
            return cv2.imread(self.data[idx], cv2.IMREAD_COLOR)