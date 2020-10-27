import tensorflow as tf
import re
import numpy as np
from config.load_config import config
from augmentations import transform


def read_labeled_tfrecord(example):
    tfrec_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "patient_id": tf.io.FixedLenFeature([], tf.int64),
        "sex": tf.io.FixedLenFeature([], tf.int64),
        "age_approx": tf.io.FixedLenFeature([], tf.int64),
        "anatom_site_general_challenge": tf.io.FixedLenFeature([], tf.int64),
        "diagnosis": tf.io.FixedLenFeature([], tf.int64),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return example["image"], example["target"]


def read_unlabeled_tfrecord(example, return_image_name):
    tfrec_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return example["image"], example["image_name"] if return_image_name else 0


def prepare_image(img, augment=True, config=None):
    img = tf.image.decode_jpeg(img, channels=3)
    # Cast and normalize the image to [0,1]
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        img = transform(img, config)
        img = tf.image.random_crop(
            img,
            [config["CROP_SIZE"][config["IMG_SIZES"]], config["CROP_SIZE"][config["IMG_SIZES"]], 3],
        )
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_hue(img, 0.01)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)
        # resize is needed as we random cropped
        img = tf.image.resize(img, [config["IMG_SIZES"], config["IMG_SIZES"]])
        # I am not sure why values will go outside of the stipulated range of 0,1
        img = tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)
    # investigate why this will prompt error in shapes when I run enumerate(iter(ds))

    # else:
    #     img = tf.image.central_crop(
    #         img, config["CROP_SIZE"][config["IMG_SIZES"]] / config["IMG_SIZES"]
    #     )

    img = tf.reshape(img, [config["IMG_SIZES"], config["IMG_SIZES"], 3])

    return img


# function to count how many photos we have in
# the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
# note that if you are using 1 single tfrec then this code will not work because this assumes filenames is a list of tfrec
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

