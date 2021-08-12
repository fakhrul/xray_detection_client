"""
Client side code to perform a single API call to a tensorflow model up and running.
"""
import argparse
import json

import numpy as np
import requests
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import plot_util
from object_detection.utils import label_map_util
import object_detection.utils.ops as utils_ops
from PIL import Image

import os
import json, requests
from flask import Flask, request, jsonify, send_from_directory
app = Flask(__name__)

SERVER_URL = 'http://localhost:8501/v1/models/my_model:predict'
PATH_TO_LABELS = './xray/labels.pbtxt'

@app.route('/')
def hello_world():
    return 'X-RAY Bone Fracture Detection API v1.0'


@app.route('/image-quality', methods=['POST'])
def image_quality():
    data = {}

    if not request.files["image"]:
        return jsonify({"status": 400, "message": 'No image passed'})

    img = request.files["image"]
    
    img.save(img.filename)
    RAW_FILE = img.filename

    output_filename = process(filename=img.filename)

    print(output_filename)
    print(os.getcwd())

    full_output_path = os.path.join(os.getcwd(), output_filename)
    print(full_output_path)
    # if os.path.exists(full_output_path):
    #     return 'sucess'
    # else:
    #     return ' failed'
    return send_from_directory(os.getcwd(), output_filename)

def format_mask(detection_masks, detection_boxes, N, image_size):
    """
    Format the m*m detection soft masks as full size binary masks. 

    Args:
        detection_masks (np.array): of size N * m * m
        detection_boxes (np.array): of size N * 4 with the normalized bow coordinates.
            Coordinates are written as [y_min, x_min, y_max, x_max]
        N (int): number of detections in the image
        image_size (tuple(int))

    Returns:
        detection_masks (np.array): of size N * H * W  where H and W are the image Height and Width.
    
    """
    (height, width, _) = image_size
    output_masks = np.zeros((N, image_size[0], image_size[1]))
    # Process the masks related to the N objects detected in the image
    for i in range(N):
        normalized_mask = detection_masks[i].astype(np.float32)
        normalized_mask = Image.fromarray(normalized_mask, 'F')

        # Boxes are expressed with 4 scalars - normalized coordinates [y_min, x_min, y_max, x_max]
        [y_min, x_min, y_max, x_max] = detection_boxes[i]

        # Compute absolute boundary of box
        box_size = (int((x_max - x_min) * width), int((y_max - y_min) * height)) 

        # Resize the mask to the box size using LANCZOS appoximation
        resized_mask = normalized_mask.resize(box_size, Image.LANCZOS)
        
        # Convert back to array
        resized_mask = np.array(resized_mask).astype(np.float32)

        # Binarize the image by using a fixed threshold
        binary_mask_box = np.zeros(resized_mask.shape)
        thresh = 0.5
        (h, w) = resized_mask.shape

        for k in range(h):
            for j in range(w):
                if resized_mask[k][j] >= thresh:
                    binary_mask_box[k][j] = 1

        binary_mask_box = binary_mask_box.astype(np.uint8)

        # Replace the mask in the context of the original image size
        binary_mask = np.zeros((height, width))
        
        x_min_at_scale = int(x_min * width)
        y_min_at_scale = int(y_min * height)

        d_x = int((x_max - x_min) * width)
        d_y = int((y_max - y_min) * height)

        for x in range(d_x):
            for y in range(d_y):
                binary_mask[y_min_at_scale + y][x_min_at_scale + x] = binary_mask_box[y][x] 
        
        # Update the masks array
        output_masks[i][:][:] = binary_mask

    # Cast mask as integer
    output_masks = output_masks.astype(np.uint8)
    return output_masks

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def pre_process(image_path):
    """
    Pre-process the input image to return a json to pass to the tf model

    Args:
        image_path (str):  Path to the jpeg image

    Returns:
        formatted_json_input (str)
    """

    image = Image.open(image_path).convert("RGB")
    image_np = plot_util.load_image_into_numpy_array(image)

    # Expand dims to create  bach of size 1
    image_tensor = np.expand_dims(image_np, 0)
    formatted_json_input = json.dumps({"signature_name": "serving_default", "instances": image_tensor.tolist()})

    return formatted_json_input


def post_process(server_response, image_size):
    """
    Post-process the server response

    Args:
        server_response (requests.Response)
        image_size (tuple(int))

    Returns:
        post_processed_data (dict)
    """
    response = json.loads(server_response.text)
    output_dict = response['predictions'][0]

    # all outputs are float32 numpy arrays, so convert types as appropriate

    output_dict['num_detections'] = int(output_dict['num_detections'])
    output_dict['detection_classes'] = np.array([int(class_id) for class_id in output_dict['detection_classes']])
    output_dict['detection_boxes'] = np.array(output_dict['detection_boxes'])
    output_dict['detection_scores'] = np.array(output_dict['detection_scores'])

    # Process detection mask
    if 'detection_masks' in output_dict:
        # Determine a threshold above wihc we consider the pixel shall belong to the mask
        # thresh = 0.5
        output_dict['detection_masks'] = np.array(output_dict['detection_masks'])
        output_dict['detection_masks'] = format_mask(output_dict['detection_masks'], output_dict['detection_boxes'], output_dict['num_detections'], image_size)
    
    return output_dict


def process(filename):
    image_path = os.path.join(os.getcwd(), filename)
    print('image_path', image_path)
    # image_path = args.image_path
    output_image = os.path.join(os.getcwd(), 'OUTPUT.JSON')
    # save_output_image = os.path.join(os.getcwd(), 'OUTPUT.PNG')
    # path_to_labels = args.label_map

    # Build input data
    print(f'\n\nPre-processing input file {image_path}...\n')
    formatted_json_input = pre_process(image_path)
    print('Pre-processing done! \n')

    # Call tensorflow server
    headers = {"content-type": "application/json"}
    print(f'\n\nMaking request to {SERVER_URL}...\n')
    server_response = requests.post(SERVER_URL, data=formatted_json_input, headers=headers)
    print(f'Request returned\n')

    # Post process output
    print(f'\n\nPost-processing server response...\n')
    image = Image.open(image_path).convert("RGB")
    image_np = load_image_into_numpy_array(image)
    output_dict = post_process(server_response, image_np.shape)
    print(f'Post-processing done!\n')

    # Save output on disk
    print(f'\n\nSaving output to {output_image}\n\n')
    with open(output_image, 'w+') as outfile:
        json.dump(json.loads(server_response.text), outfile)
    print(f'Output saved!\n')

    # if save_output_image:
    # Save output on disk
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8,
        )
    output_with_no_extension = output_image.split('.', 1)[0]
    output_image = ''.join([output_with_no_extension, '.jpeg'])
    print(output_image)
    img = Image.fromarray(image_np)
    img.save(output_image)
    print('\n\nImage saved\n\n')
    return 'OUTPUT.jpeg'


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)