import time
import os
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util import nest

flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('tfliteoutput', './lite/yolov3.tflite', 'path to TFLite model')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    if FLAGS.tiny:
        yolo = YoloV3Tiny(size=416)
        #yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(size=416)
        #yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    try:
        os.mkdir('lite')
    except FileExistsError:
        pass

    yolo.save('lite/yolov3_inference.hdf5')
    with open('lite/yolov3_inference.json', 'w') as f:
        jsonstr = yolo.to_json(indent=4)
        f.write(jsonstr)

    print("Model saved, converting to TFLite")
    converter = tf.lite.TFLiteConverter.from_keras_model(yolo)
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    logging.info("model converted")
    open(FLAGS.tfliteoutput, "wb").write(tflite_model)

    logging.info("model saved to: {}".format(FLAGS.tfliteoutput))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
