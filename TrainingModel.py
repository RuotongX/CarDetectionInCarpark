import os
import pathlib


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

picturepath = 'model/car_ims/'
imagesize = [0,0]

def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    imagesize[0] = im_width
    imagesize[1] = im_height
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
    image_np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.8)
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)

train_image_dir = picturepath
train_images_np = []
# gt_str = ''

annots = loadmat('model/carbox.mat')
type(annots['annotations']),annots['annotations'].shape
type(annots['annotations'][0][0]),annots['annotations'][0][0].shape
[item.flat[0] for item in annots['annotations'][0][0]]
data = [[row.flat[0] for row in line] for line in annots['annotations'][0]]

for i in range(100,229):
    image_path = os.path.join(train_image_dir, '000' + str(i) + '.jpg')
    train_images_np.append(load_image_into_numpy_array(image_path))
    # box = 'np.array([[' + str(format(data[i - 1][1]/imagesize[0], '.3f')) + ',' + str(format(data[i - 1][2]/imagesize[1], '.3f')) + ',' + str(format(data[i - 1][3]/imagesize[0], '.3f')) + ',' + str(
    #     format(data[i - 1][4]/imagesize[1], '.3f')) + ']], dtype=np.float32),'
    # gt_str = gt_str + box

# print(gt_str[:-1])
gt_boxes = [np.array([[0.067,0.071,0.956,0.956]], dtype=np.float32),np.array([[0.067,0.072,0.912,0.885]], dtype=np.float32),np.array([[0.035,0.383,0.967,0.897]], dtype=np.float32),np.array([[0.113,0.323,0.900,0.848]], dtype=np.float32),np.array([[0.199,0.328,0.837,0.837]], dtype=np.float32),np.array([[0.142,0.258,0.917,0.881]], dtype=np.float32),np.array([[0.174,0.322,0.863,0.807]], dtype=np.float32),np.array([[0.080,0.354,0.819,0.906]], dtype=np.float32),np.array([[0.027,0.203,0.980,0.824]], dtype=np.float32),np.array([[0.120,0.200,0.870,0.870]], dtype=np.float32),np.array([[0.135,0.029,0.874,0.997]], dtype=np.float32),np.array([[0.036,0.279,0.967,0.727]], dtype=np.float32),np.array([[0.056,0.042,0.931,0.973]], dtype=np.float32),np.array([[0.040,0.227,0.968,0.840]], dtype=np.float32),np.array([[0.040,0.210,0.989,0.890]], dtype=np.float32),np.array([[0.087,0.198,0.980,0.767]], dtype=np.float32),np.array([[0.097,0.348,0.899,0.855]], dtype=np.float32),np.array([[0.176,0.134,0.827,0.891]], dtype=np.float32),np.array([[0.094,0.213,0.891,0.793]], dtype=np.float32),np.array([[0.043,0.050,0.959,0.932]], dtype=np.float32),np.array([[0.086,0.335,0.955,0.876]], dtype=np.float32),np.array([[0.029,0.336,0.960,0.874]], dtype=np.float32),np.array([[0.135,0.266,0.844,0.786]], dtype=np.float32),np.array([[0.089,0.124,0.761,0.934]], dtype=np.float32),np.array([[0.219,0.163,0.957,0.753]], dtype=np.float32),np.array([[0.037,0.115,0.919,0.844]], dtype=np.float32),np.array([[0.042,0.182,0.966,0.889]], dtype=np.float32),np.array([[0.101,0.020,0.925,0.968]], dtype=np.float32),np.array([[0.053,0.127,0.949,0.926]], dtype=np.float32),np.array([[0.100,0.055,0.897,0.985]], dtype=np.float32),np.array([[0.102,0.153,0.882,0.868]], dtype=np.float32),np.array([[0.291,0.315,0.717,0.723]], dtype=np.float32),np.array([[0.080,0.245,0.911,0.961]], dtype=np.float32),np.array([[0.040,0.166,0.973,0.859]], dtype=np.float32),np.array([[0.007,0.462,1.000,0.902]], dtype=np.float32),np.array([[0.056,0.249,0.983,0.938]], dtype=np.float32),np.array([[0.097,0.275,0.930,0.885]], dtype=np.float32),np.array([[0.025,0.280,0.957,0.854]], dtype=np.float32),np.array([[0.114,0.233,0.806,0.850]], dtype=np.float32),np.array([[0.087,0.286,0.940,0.769]], dtype=np.float32),np.array([[0.085,0.447,0.935,0.936]], dtype=np.float32),np.array([[0.117,0.214,0.892,0.882]], dtype=np.float32),np.array([[0.098,0.283,0.885,0.777]], dtype=np.float32),np.array([[0.126,0.047,0.841,0.884]], dtype=np.float32),np.array([[0.019,0.176,0.986,0.925]], dtype=np.float32),np.array([[0.022,0.374,0.991,0.946]], dtype=np.float32),np.array([[0.066,0.106,0.934,0.851]], dtype=np.float32),np.array([[0.134,0.461,0.841,0.906]], dtype=np.float32),np.array([[0.096,0.404,0.904,0.846]], dtype=np.float32),np.array([[0.117,0.297,0.817,0.858]], dtype=np.float32),np.array([[0.019,0.245,0.826,0.755]], dtype=np.float32),np.array([[0.123,0.149,0.973,0.941]], dtype=np.float32),np.array([[0.108,0.498,0.907,0.959]], dtype=np.float32),np.array([[0.047,0.086,0.967,0.864]], dtype=np.float32),np.array([[0.097,0.401,0.903,0.898]], dtype=np.float32),np.array([[0.050,0.180,0.965,0.789]], dtype=np.float32),np.array([[0.039,0.124,0.981,0.814]], dtype=np.float32),np.array([[0.016,0.215,0.990,0.958]], dtype=np.float32),np.array([[0.040,0.172,0.964,0.870]], dtype=np.float32),np.array([[0.059,0.369,0.918,0.889]], dtype=np.float32),np.array([[0.230,0.260,0.736,0.887]], dtype=np.float32),np.array([[0.036,0.224,0.956,0.819]], dtype=np.float32),np.array([[0.073,0.250,0.942,0.927]], dtype=np.float32),np.array([[0.218,0.311,0.895,0.825]], dtype=np.float32),np.array([[0.055,0.309,0.954,0.930]], dtype=np.float32),np.array([[0.135,0.377,0.882,0.805]], dtype=np.float32),np.array([[0.087,0.161,0.929,0.933]], dtype=np.float32),np.array([[0.111,0.198,0.915,0.906]], dtype=np.float32),np.array([[0.059,0.198,0.921,0.879]], dtype=np.float32),np.array([[0.083,0.301,0.923,0.820]], dtype=np.float32),np.array([[0.016,0.219,0.984,0.837]], dtype=np.float32),np.array([[0.092,0.266,0.938,0.896]], dtype=np.float32),np.array([[0.104,0.146,0.974,0.784]], dtype=np.float32),np.array([[0.094,0.190,0.919,0.777]], dtype=np.float32),np.array([[0.031,0.417,0.954,0.958]], dtype=np.float32),np.array([[0.073,0.362,0.934,0.866]], dtype=np.float32),np.array([[0.052,0.147,0.954,0.865]], dtype=np.float32),np.array([[0.185,0.227,0.781,0.706]], dtype=np.float32),np.array([[0.122,0.253,0.888,0.788]], dtype=np.float32),np.array([[0.075,0.034,0.957,0.940]], dtype=np.float32),np.array([[0.090,0.268,0.905,0.860]], dtype=np.float32),np.array([[0.002,0.177,0.998,0.911]], dtype=np.float32),np.array([[0.093,0.353,0.952,0.908]], dtype=np.float32),np.array([[0.022,0.175,0.980,0.829]], dtype=np.float32),np.array([[0.010,0.209,0.987,0.813]], dtype=np.float32),np.array([[0.087,0.142,0.928,0.758]], dtype=np.float32),np.array([[0.130,0.208,0.871,0.809]], dtype=np.float32),np.array([[0.050,0.077,0.927,0.934]], dtype=np.float32),np.array([[0.055,0.401,0.950,0.933]], dtype=np.float32),np.array([[0.001,0.001,0.927,0.843]], dtype=np.float32),np.array([[0.080,0.119,0.922,0.838]], dtype=np.float32),np.array([[0.111,0.400,0.928,0.969]], dtype=np.float32),np.array([[0.177,0.248,0.878,0.816]], dtype=np.float32),np.array([[0.031,0.022,0.975,0.976]], dtype=np.float32),np.array([[0.124,0.028,0.888,0.966]], dtype=np.float32),np.array([[0.028,0.179,0.964,0.805]], dtype=np.float32),np.array([[0.143,0.041,0.956,0.978]], dtype=np.float32),np.array([[0.060,0.431,0.940,0.902]], dtype=np.float32),np.array([[0.013,0.387,0.997,0.904]], dtype=np.float32),np.array([[0.212,0.320,0.788,0.918]], dtype=np.float32),np.array([[0.043,0.173,0.940,0.963]], dtype=np.float32),np.array([[0.079,0.242,0.935,0.941]], dtype=np.float32),np.array([[0.031,0.164,0.974,0.816]], dtype=np.float32),np.array([[0.052,0.008,0.999,0.858]], dtype=np.float32),np.array([[0.016,0.071,0.987,0.968]], dtype=np.float32),np.array([[0.053,0.414,0.917,0.869]], dtype=np.float32),np.array([[0.041,0.306,0.963,0.869]], dtype=np.float32),np.array([[0.071,0.211,0.849,0.561]], dtype=np.float32),np.array([[0.146,0.123,0.844,0.889]], dtype=np.float32),np.array([[0.030,0.325,0.930,0.975]], dtype=np.float32),np.array([[0.004,0.349,1.000,0.968]], dtype=np.float32),np.array([[0.026,0.145,0.963,0.902]], dtype=np.float32),np.array([[0.164,0.072,0.952,0.930]], dtype=np.float32),np.array([[0.133,0.344,0.803,0.818]], dtype=np.float32),np.array([[0.057,0.225,0.973,0.892]], dtype=np.float32),np.array([[0.072,0.204,0.968,0.967]], dtype=np.float32),np.array([[0.170,0.240,0.820,0.827]], dtype=np.float32),np.array([[0.066,0.273,0.888,0.881]], dtype=np.float32),np.array([[0.128,0.279,0.879,0.914]], dtype=np.float32),np.array([[0.170,0.253,0.891,0.800]], dtype=np.float32),np.array([[0.111,0.341,0.899,0.876]], dtype=np.float32),np.array([[0.108,0.261,0.979,0.844]], dtype=np.float32),np.array([[0.053,0.045,0.963,0.975]], dtype=np.float32),np.array([[0.151,0.236,0.843,0.966]], dtype=np.float32),np.array([[0.187,0.236,0.833,0.853]], dtype=np.float32),np.array([[0.085,0.080,0.888,0.960]], dtype=np.float32),np.array([[0.164,0.135,0.829,0.908]], dtype=np.float32),np.array([[0.092,0.161,0.810,0.867]], dtype=np.float32),np.array([[0.118,0.090,0.882,0.962]], dtype=np.float32)
    ]
#

plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labelsize'] = False
plt.rcParams['ytick.labelsize'] = False
plt.rcParams['xtick.top'] = False
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.right'] = False
plt.rcParams['figure.figsize'] = [14, 7]

# for idx, train_image_np in enumerate(train_images_np):
#   plt.subplot(13, 10, idx+1)
#   plt.imshow(train_image_np)
# plt.show()

car_class_id = 1
num_classes = 1

category_index = {car_class_id: {'id': car_class_id, 'name': 'car'}}

# Convert class labels to one-hot; convert everything to tensors.
# The `label_id_offset` here shifts all classes by a certain number of indices;
# we do this here so that the model receives one-hot labels where non-background
# classes start counting at the zeroth index.  This is ordinarily just handled
# automatically in our training binaries, but we need to reproduce it here.
label_id_offset = 1
train_image_tensors = []
gt_classes_one_hot_tensors = []
gt_box_tensors = []
for (train_image_np, gt_box_np) in zip(
    train_images_np, gt_boxes):
  train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(
      train_image_np, dtype=tf.float32), axis=0))
  gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
  zero_indexed_groundtruth_classes = tf.convert_to_tensor(
      np.ones(shape=[gt_box_np.shape[0]], dtype=np.int32) - label_id_offset)
  gt_classes_one_hot_tensors.append(tf.one_hot(
      zero_indexed_groundtruth_classes, num_classes))
print('Done prepping data.')

dummy_scores = np.array([1.0], dtype=np.float32)  # give boxes a score of 100%

plt.figure(figsize=(30, 15))
for idx in range(129):
  plt.subplot(13, 10, idx+1)
  plot_detections(
      train_images_np[idx],
      gt_boxes[idx],
      np.ones(shape=[gt_boxes[idx].shape[0]], dtype=np.int32),
      dummy_scores, category_index)
plt.show()

tf.keras.backend.clear_session()

print('Building model and restoring weights for fine-tuning...', flush=True)
num_classes = 1
pipeline_config = 'efficientdet_d3_coco17_tpu-32/pipeline.config'
checkpoint_path = 'efficientdet_d3_coco17_tpu-32/checkpoint/ckpt-0.index'

# Load pipeline config and build a detection model.
#
# Since we are working off of a COCO architecture which predicts 90
# class slots by default, we override the `num_classes` field here to be just
# one (for our new rubber ducky class).
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
model_config.ssd.num_classes = num_classes
model_config.ssd.freeze_batchnorm = True
detection_model = model_builder.build(
      model_config=model_config, is_training=True)

# Set up object-based checkpoint restore --- RetinaNet has two prediction
# `heads` --- one for classification, the other for box regression.  We will
# restore the box regression head but initialize the classification head
# from scratch (we show the omission below by commenting out the line that
# we would add if we wanted to restore both heads)
fake_box_predictor = tf.compat.v2.train.Checkpoint(
    _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
    # _prediction_heads=detection_model._box_predictor._prediction_heads,
    #    (i.e., the classification head that we *will not* restore)
    _box_prediction_head=detection_model._box_predictor._box_prediction_head,
    )
fake_model = tf.compat.v2.train.Checkpoint(
          _feature_extractor=detection_model._feature_extractor,
          _box_predictor=fake_box_predictor)
ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
ckpt.restore(checkpoint_path).expect_partial()

# Run model through a dummy image so that variables are created
image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
prediction_dict = detection_model.predict(image, shapes)
_ = detection_model.postprocess(prediction_dict, shapes)
print('Weights restored!')


tf.keras.backend.set_learning_phase(True)

# These parameters can be tuned; since our training set has 5 images
# it doesn't make sense to have a much larger batch size, though we could
# fit more examples in memory if we wanted to.
batch_size = 4
learning_rate = 0.01
num_batches = 100

# Select variables in top layers to fine-tune.
trainable_variables = detection_model.trainable_variables
to_fine_tune = []
prefixes_to_train = [
  'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
  'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
for var in trainable_variables:
  if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
    to_fine_tune.append(var)

# Set up forward + backward pass for a single train step.
def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
  """Get a tf.function for training step."""

  # Use tf.function for a bit of speed.
  # Comment out the tf.function decorator if you want the inside of the
  # function to run eagerly.
  @tf.function
  def train_step_fn(image_tensors,
                    groundtruth_boxes_list,
                    groundtruth_classes_list):
    """A single training iteration.

    Args:
      image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
        Note that the height and width can vary across images, as they are
        reshaped within this function to be 640x640.
      groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
        tf.float32 representing groundtruth boxes for each image in the batch.
      groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
        with type tf.float32 representing groundtruth boxes for each image in
        the batch.

    Returns:
      A scalar tensor representing the total loss for the input batch.
    """
    shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
    model.provide_groundtruth(
        groundtruth_boxes_list=groundtruth_boxes_list,
        groundtruth_classes_list=groundtruth_classes_list)
    with tf.GradientTape() as tape:
      preprocessed_images = tf.concat(
          [detection_model.preprocess(image_tensor)[0]
           for image_tensor in image_tensors], axis=0)
      prediction_dict = model.predict(preprocessed_images, shapes)
      losses_dict = model.loss(prediction_dict, shapes)
      total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
      gradients = tape.gradient(total_loss, vars_to_fine_tune)
      optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
    return total_loss

  return train_step_fn

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
train_step_fn = get_model_train_step_function(
    detection_model, optimizer, to_fine_tune)

print('Start fine-tuning!', flush=True)
for idx in range(num_batches):
  # Grab keys for a random subset of examples
  all_keys = list(range(len(train_images_np)))
  random.shuffle(all_keys)
  example_keys = all_keys[:batch_size]

  # Note that we do not do data augmentation in this demo.  If you want a
  # a fun exercise, we recommend experimenting with random horizontal flipping
  # and random cropping :)
  gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
  gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
  image_tensors = [train_image_tensors[key] for key in example_keys]

  # Training step (forward pass + backwards pass)
  total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

  if idx % 10 == 0:
    print('batch ' + str(idx) + ' of ' + str(num_batches)
    + ', loss=' +  str(total_loss.numpy()), flush=True)

print('Done fine-tuning!')

test_image_dir = picturepath
test_images_np = []
for i in range(230,240):
    image_path = os.path.join(test_image_dir, '000' + str(i) + '.jpg')
    test_images_np.append(np.expand_dims(
        load_image_into_numpy_array(image_path), axis=0))


    # Again, uncomment this decorator if you want to run inference eagerly
    @tf.function
    def detect(input_tensor):
        """Run detection on an input image.

        Args:
          input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
            Note that height and width can be anything since the image will be
            immediately resized according to the needs of the model within this
            function.

        Returns:
          A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
            and `detection_scores`).
        """
        preprocessed_image, shapes = detection_model.preprocess(input_tensor)
        prediction_dict = detection_model.predict(preprocessed_image, shapes)
        return detection_model.postprocess(prediction_dict, shapes)


    # Note that the first frame will trigger tracing of the tf.function, which will
    # take some time, after which inference should be fast.

    label_id_offset = 1
    for i in range(len(test_images_np)):
        input_tensor = tf.convert_to_tensor(test_images_np[i], dtype=tf.float32)
        detections = detect(input_tensor)

        plot_detections(
            test_images_np[i][0],
            detections['detection_boxes'][0].numpy(),
            detections['detection_classes'][0].numpy().astype(np.uint32)
            + label_id_offset,
            detections['detection_scores'][0].numpy(),
            category_index, figsize=(15, 20), image_name="gif_frame_" + ('%02d' % i) + ".jpg")

    imageio.plugins.freeimage.download()

    anim_file = 'duckies_test.gif'

    filenames = glob.glob('gif_frame_*.jpg')
    filenames = sorted(filenames)
    last = -1
    images = []
    for filename in filenames:
        image = imageio.imread(filename)
        images.append(image)

    imageio.mimsave(anim_file, images, 'GIF-FI', fps=5)

    display(IPyImage(open(anim_file, 'rb').read()))