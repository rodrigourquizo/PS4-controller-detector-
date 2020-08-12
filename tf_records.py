import tensorflow as tf
import pandas as pd
from collections import namedtuple
import os
from object_detection.utils import dataset_util

csv_folder = "./csv"
records_folder = "./training/records"
resized_images = "./data"
train_resized_images = os.path.join(resized_images , "train")
test_resized_images = os.path.join(resized_images , "test")


def get_groups(path):
    df = pd.read_csv(path)
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby("filename")
    groups = []
    for x in gb.groups:
        frame = gb.get_group(x)
        obj = data(x,frame)
        groups.append(obj)
    
    return groups

def encode_class(name):
    if name == "controller":
        return 1
    else:
        return None
    
def create_tf_example(group, path_images):    
    frame = group.object
    image = open(os.path.join(path_images, group.filename), "rb").read()
    image_format = b'jpg'
    xmins = list(frame["xmin"].values.astype(float))
    ymins = list(frame["ymin"].values.astype(float))
    xmaxs = list(frame["xmax"].values.astype(float))
    ymaxs = list(frame["ymax"].values.astype(float))
    classes_text = list(frame["class"].values.astype(bytes))
    classes = list(frame["class"].apply(encode_class).values.astype(int))
    filename = group.filename.encode("utf8")
    feature = {
        'image/height': dataset_util.int64_feature(300),
        'image/width': dataset_util.int64_feature(300),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example

def main(path_csv, path_images, name):
    writer = tf.io.TFRecordWriter(os.path.join(records_folder, name))
    for group in get_groups(path_csv):
        tf_example = create_tf_example(group, path_images)
        writer.write(tf_example.SerializeToString())
    writer.close()    
        
if __name__ == "__main__":
    main(os.path.join(csv_folder, "train.csv"), train_resized_images, "train.record")
    main(os.path.join(csv_folder, "test.csv"), test_resized_images, "test.record")
        