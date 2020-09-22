import numpy as np
import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
label_map_util.tf = tf.compat.v1
tf.gfile = tf.io.gfile
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils


# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5500)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model    efficientdet_d3_coco17_tpu-32
PATH_TO_CKPT = 'efficientdet_d3_coco17_tpu-32/checkpoint/'
PATH_TO_CFG = 'efficientdet_d3_coco17_tpu-32/pipeline.config'
PATH_TO_SAVED_MODEL = 'efficientdet_d3_coco17_tpu-32/saved_model'

PATH_TO_LABELS = 'faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8/mscoco_label_map.pbtxt'
cap = cv2.VideoCapture('test1.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps =cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('out.avi', fourcc, fps, (640, 256))

detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)



while True:
    # Read frame from camera
    ret, image_np = cap.read()

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)


    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = 100
    label_id_offset = 1
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    #detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()

    # print(detections)
    detectionlist = []
    for i in range(len(detections['detection_boxes'])):
        box = detections['detection_boxes'][i]
        Itemclass = detections['detection_classes'][i].astype(np.int64)
        score = detections['detection_scores'][i]
        detectionItem = [box,Itemclass,score]
        detectionlist.append(detectionItem)

    carlist = []
    for i in range(len(detectionlist)):
        if detectionlist[i][1] == 3 or detectionlist[i][1] == 6 and detectionlist[i][2] > 0.25:
            y = (detectionlist[i][0][0]*256+detectionlist[i][0][2]*256)/2
            x = (detectionlist[i][0][1]*640+detectionlist[i][0][3]*640)/2
            car = []
            car.append(x)
            car.append(y)
            carlist.append(car)

    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #             image_np_with_detections,
    #             detections['detection_boxes'],
    #             detections['detection_classes'],
    #             detections['detection_scores'],
    #             category_index,
    #             use_normalized_coordinates=True,
    #             max_boxes_to_draw=70,
    #             min_score_thresh=.40,
    #             agnostic_mode=False)
    # print(len(carlist))

    parkingbox = [[3,72,46,225,0],[52,77,120,224,0],[135,76,202,241,0],[218,81,282,241,0],[297,62,364,242,0],[376,81,455,241,0],[463,89,542,238,0],[547,89,636,244,0]]

    for park in parkingbox:
        for car in carlist:
            if park[0]+10 < car[0] < park[2]-10 and park[1]+10 < car[1] < park[3]-10:
                park[4] = 1




    for park in parkingbox:
        if park[4] == 1:
            cv2.rectangle(image_np_with_detections,(park[0],park[1]),(park[2],park[3]),(0,0,255),2)
        else:
            cv2.rectangle(image_np_with_detections,(park[0],park[1]),(park[2],park[3]),(255,255,255),2)

    if ret == True:
        frame = cv2.flip(image_np_with_detections,0)

        out.write(image_np_with_detections)
        # Display output
        cv2.imshow('object detection', cv2.resize(image_np_with_detections, (640, 256)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
