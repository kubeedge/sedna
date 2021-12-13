import pickle
import requests
import cv2

from sedna.common.log import LOGGER
from sedna.common.benchmark import FTimer

def write_text_on_image(img, text, textX, textY):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 1, 2)[0]

    # get coords based on boundary
    # textX = (img.shape[1] - textsize[0]) / 2
    # textY = (img.shape[0] + textsize[1]) / 2

    # add text centered on image
    cv2.putText(img, text, (textX, textY), font, 1, (0, 255, 0), 2)

def transfer_reid_result(result, endpoint, post_process=False):
    """
    Send to the backend endpoint the ReID result.

    With post_process set to False, it sends out a DetTrackResult object. This means that 
    the receiving backend knows how to unpickle such object.
    
    With post_process set to True, it sends out an encoded image overlayed with bounding boxes,
    camera ID, and object IDs.
    """
    try:
        LOGGER.debug("Transferring final result to backend")
        with FTimer(f"upload_fe_reid_results"):
            if not post_process:
                # Output: Pickled DetTrackResult Object
                status = requests.post(endpoint, pickle.dumps(result))
            else:
                # Output: JPG encoded image
                status = requests.post(endpoint, prepare_output(result))
        LOGGER.debug(status.status_code)
    except Exception as ex:
        LOGGER.error(f'Unable to upload reid results to backend. [{ex}]')


def prepare_output(data):
    try:
        image = cv2.imdecode(data.scene, cv2.IMREAD_COLOR)

        for idx, bbox in enumerate(data.bbox_coord):
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255, 0), 2)
        
            # Add ID
            write_text_on_image(image, data.ID[0], bbox[0], bbox[1]-10)
           
        # Add camera
        write_text_on_image(image, f"Camera:{data.camera[idx]}", 0, 30)

        output = cv2.imencode('.jpg', image)[1]
        return output

    except Exception as ex:
        LOGGER.error(f"Error during output scene preparation. {[ex]}")
        return None
    