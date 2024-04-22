import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


# Set TensorFlow logging level to suppress unnecessary messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the model
PATH_TO_MODEL_DIR = r'f:\My__model\saved_model'
PATH_TO_LABELS = r'F:\label_map.pbtxt'
MIN_CONF_THRESH = 0.5

detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)

# Load label map data
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)

    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    frame_with_detections = frame.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=MIN_CONF_THRESH,
        agnostic_mode=False
    )

    cv2.imshow('Object Detection', frame_with_detections)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
