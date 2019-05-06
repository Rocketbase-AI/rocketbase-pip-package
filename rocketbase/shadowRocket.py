import os

from PIL import Image
from PIL import ImageDraw

import requests

import rocketbase.exceptions

class ShadowRocket:
    """ A ShadowRocket is a Rocket placeholder to use the Rocket's API

    In order to use seemlessly the Deep Learning models on the machine and the Cloud API we use a ShadowRocket instead of the usual Rocket. The ShadowRocket has the same functions as the normal Rocket but they work differently to allow the use of the API instead of on-device deep learning model.

    Attributes:
        rocket_info (dict): It contains all of the information relative to Rocket including the apiUrl and the family of the Rocket.
    """

    def __init__(self, rocket_info: dict):
        """ Initiate the ShadowRocket

        Args:
            rocket_info (dict): Information relative to the Rocket. It needs to contain apiUrl and the family of the Rocket.

        Raises:
            RocketNotEnoughInfo: If not enough information is provided in rocket_info
        """
        # List of all the required information to create the ShadowRocket
        LIST_REQUIRED_INFO = [
            'apiUrl',
            'family'
        ]

        # Check if all the needed information are provided
        missing_info = set(LIST_REQUIRED_INFO) - rocket_info.keys()
        empty_info = [k for k, i in rocket_info.items() if not isinstance(
            i, bool) and not i and k in LIST_REQUIRED_INFO]
        
        if missing_info or empty_info:
            raise rocketbase.exceptions.RocketNotEnoughInfo(
                'Missing the following information to create the ShadowRocket: {}.'.format(', '.join(set(missing_info + empty_info)))
            )
        
        # Keep the rocket_info
        self.rocket_info = rocket_info

    def eval(self):
        """ Placeholder for when the eval function is called on the ShadowRocket

        Returns:
            self: Indeed in certain cases, it is needed that the eval function returns the ShadowRocket itself.
        """
        return self
    
    def preprocess(self, img: Image):
        """ Prepare the image to send it to the API 

        Args:
            img (PIL.Image): image to process with the API
        
        Returns:
            PATH_TEMP_IMG (str): path to the temporary image saved to the disk
        """
        # Path to the temporary image
        PATH_TEMP_IMG = '.temp.png'

        # save the image to the disk
        img.save(PATH_TEMP_IMG)

        return PATH_TEMP_IMG
    
    def __call__(self, img_path: str):
        """ Use the API to simulate the pass-forward of the model

        Args:
            img_path (str): path to the image to process

        Returns:
            The json answer from the API. 
        """
        # Do the pass-forward using the API
        r = requests.post(self.rocket_info['apiUrl'], files=dict(input=open(img_path, 'rb')))

        # Delete the temp image
        os.remove(img_path)

        return r.json()

    def postprocess(self, api_results, input_img: Image, visualize: bool = False):
        """ Preprocess the answer from the API.
        
        In most cases the answer of the API has already been postprocess. This function is to use the information to visualize the answer from the API.

        Args:
            api_results (list[dict] or PIL.Image): either the formated output of the model or directly an image with the output visualization.
            input_img (PIL.Image): initial image analized by the model
            visualize (bool): if the function should returns the raw data or the visualization of the output.

        Returns:
            api_results (list[dict]) / img_out (PIL.Image): Either the input directly or the visualization of the output.

        Raises:
            RocketInfoFormat: If the format of the api_results doesn't match the desired postprocessing method. (e.g. if api_results is only an image but the user wants the list of the detections.)
        """
        if visualize and isinstance(api_results, Image.Image):
            return api_results
        
        elif not visualize and isinstance(api_results, list):
            return api_results

        elif visualize and isinstance(api_results, list):
            line_width = 2
            img_out = input_img
            ctx = ImageDraw.Draw(img_out, 'RGBA')
            for detection in api_results:
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

        elif not visualize and isinstance(api_results, Image):
            raise rocketbase.exceptions.RocketInfoFormat(
                'Impossible to get the raw output of the ShadowRocket as only the image visualization was returned by the API.'
            )
        
        else:
            raise rocketbase.exceptions.RocketInfoFormat(
                'Format of the API answer is not recognized: \'{}\'.'.format(type(api_results))
            )

    