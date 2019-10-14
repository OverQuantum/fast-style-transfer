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
    parser.add_argument("--prefix", default="")
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--png", action='store_true')
    args = parser.parse_args()

    #block GPU
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    transformer = TransformerNet()
    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt.restore(tf.train.latest_checkpoint(args.log_dir)).expect_partial()

    for r, d, f in os.walk(args.image_path):
        break;

    for file in f:
        image = load_img(os.path.join(r, file))

        transformed_image = transformer(image)

        #fix issue with values out of [0,255] became wrap around
        transformed_image = tf.clip_by_value(transformed_image, 0, 255)
        
        transformed_image = tf.cast(
            tf.squeeze(transformed_image), tf.uint8
        ).numpy()

        img = Image.fromarray(transformed_image, mode="RGB")
        
        #add prefix and extension if needed
        file2 = file
        if len(args.prefix)>0:
            file2 = args.prefix + "_" + file2
        if args.png:
            file2 = file2 + ".png"
        img.save(os.path.join(args.output_path, file2))
        
        #print filename to see progress
        print(file2)
