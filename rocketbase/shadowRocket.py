import ast
import os

from PIL import Image
from PIL import ImageDraw

import requests

class ShadowRocket:

    def __init__(self, rocket_info):
        self.rocket_info = rocket_info
    
    def preprocess(self, img: Image):
        # save the image to the disk to fake it
        PATH_TEMP_IMG = 'temp.png'
        img.save(PATH_TEMP_IMG)

        return PATH_TEMP_IMG

    def eval(self):
        # need to return self other doesn't get instantiate if use:
        #   model = Rocket.land(rocket, device = 'API').eval()
        return self

    def postprocess(self, list_detections: dict, input_img: Image, visualize: bool = False):

        if visualize:
            line_width = 2
            img_out = input_img
            ctx = ImageDraw.Draw(img_out, 'RGBA')
            for detection in list_detections:
                # Extract information from the detection
                topLeft = (int(detection['topLeft_x']), int(detection['topLeft_y']))
                bottomRight = (int(detection['topLeft_x']) + int(detection['width']) - line_width, int(detection['topLeft_y']) + int(detection['height'])- line_width)
                # Sometimes some of those don't exists
                class_name = detection['class_name']
                bbox_confidence = int(detection['bbox_confidence'])
                class_confidence = int(detection['class_confidence'])

                # Draw the bounding boxes and the information related to it
                ctx.rectangle([topLeft, bottomRight], outline=(255, 0, 0, 255), width=line_width)
                ctx.text((topLeft[0] + 5, topLeft[1] + 10), text="{}, {:.2f}, {:.2f}".format(class_name, bbox_confidence, class_confidence))

            del ctx
            return img_out

        return list_detections

    def __call__(self, img_path: str):
        # Do the request
        r = requests.post(self.rocket_info['apiUrl'], files=dict(input=open(img_path, 'rb')))

        # Delete the temp image
        os.remove(img_path)

        return ast.literal_eval(r.json())

    