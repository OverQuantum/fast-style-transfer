import tensorflow as tf
import os

from PIL import Image
from argparse import ArgumentParser

from networks import TransformerNet
from utils import load_img

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log-dir")
    parser.add_argument("--image-path")
    parser.add_argument("--output-path")
    parser.add_argument("--cpu", action='store_true')
    args = parser.parse_args()

    #block GPU
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    image = load_img(args.image_path)

    transformer = TransformerNet()
    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt.restore(tf.train.latest_checkpoint(args.log_dir)).expect_partial()

    transformed_image = transformer(image)

    #fix issue with values out of [0,255] became wrap around
    transformed_image = tf.clip_by_value(transformed_image, 0, 255)
    
    transformed_image = tf.cast(
        tf.squeeze(transformed_image), tf.uint8
    ).numpy()

    img = Image.fromarray(transformed_image, mode="RGB")
    img.save(args.output_path)
