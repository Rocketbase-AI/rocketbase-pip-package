import base64
import io

import imageio
from PIL import Image

import requests

import rocketbase.exceptions


class ShadowRocket:
    """ A ShadowRocket is a Rocket placeholder to use the Rocket's API

    In order to use seemlessly the Deep Learning models on the machine and the Cloud API we use a ShadowRocket instead of the usual Rocket. The ShadowRocket has the same functions as the normal Rocket but they work differently to allow the use of the API instead of on-device deep learning model.

    In order to force the use of a ShadowRocket, add the device = 'API' flag when landing the Rocket:

        model = Rocket.land(rocket, device='API').eval()

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
        missing_info = list(set(LIST_REQUIRED_INFO) - rocket_info.keys())
        empty_info = [k for k, i in rocket_info.items() if not isinstance(
            i, bool) and not i and k in LIST_REQUIRED_INFO]

        if missing_info or empty_info:
            raise rocketbase.exceptions.RocketNotEnoughInfo(
                'Missing the following information to create the ShadowRocket: {}.'.format(
                    ', '.join(set(missing_info + empty_info)))
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
        """ Placeholder to keep the same functionalities as with the normal Rocket.

        No preprocessing is done on device when using the ShadowRocket. 

        Args:
            img (PIL.Image): image to process with the API.

        Returns:
            img (PIL.Image): return the input image unmodified.
        """
        return img


    def __call__(self, img: Image, api_visualize: bool = True):
        """ Use the API to simulate the pass-forward of the model

        Args:
            img (PIL.Image): image to process with the API.
            api_visualize (bool): Boolean to ask the API to return the visualization.

        Returns:
            The json answer from the API. 
        """
        
        # Convert the PIL.Image to a stream of Bytes to be able to send it without saving it to the disk.
        with io.BytesIO() as buffer:
            img.save(buffer, "PNG")
            buffer.seek(0)

            # Prepare the request
            files = {'input': buffer}
            values = {'visualize': 'true' if api_visualize else 'false'}

            # Do the pass-forward using the API
            r = requests.post(self.rocket_info['apiUrl'], files=files, data=values)

        return r.json()

    def postprocess(self, api_results, input_img: Image, visualize: bool = False):
        """ Preprocess the answer from the API.

        In most cases the answer of the API has already been postprocess. This function is to use the information to visualize the answer from the API.

        Args:
            api_results (list[dict]) / img_out (PIL.Image): Either the input directly or the visualization of the output.
            input_img (PIL.Image): Input image process by the model. Is here to keep the compatibility with the standard Rockets.
            visualize (bool): boolean to chose to either return the raw output or the visualization of it.

        Returns:
            image (PIL.Image): if visualize == True, returns the visualization of the output of the model.
            output (list): if visualize == False, returns the raw output of the model received from the API.

        Raises:
            RocketInfoFormat: If the format of the api_results doesn't match the desired postprocessing method. (e.g. if api_results is only an image but the user wants the list of the detections.)
        """
        if visualize:
            # Check if the visualize info is in the API answer
            if 'visualization' not in api_results.keys() or api_results['visualization'] == 'null':
                raise rocketbase.exceptions.ShadowRocketPostprocessData(
                    'Missing the visualization data from the API answer.'
                )

            # Get the b64 encoded String from the api result
            b64_str = api_results['visualization']

            # Convert the b64 string to an array containing the image
            image_array = imageio.imread(io.BytesIO(base64.b64decode(b64_str)))

            # Convert the array to the
            image = Image.fromarray(image_array)

            return image

        else:
            # Check if the visualize info is in the API answer
            if 'output' not in api_results.keys() or not api_results['output']:
                raise rocketbase.exceptions.ShadowRocketPostprocessData(
                    'Missing list of the outputs of the model from the API answer.'
                )

            return api_results['output']
