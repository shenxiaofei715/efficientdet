from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import logging
import numpy as np
import xml.etree.cElementTree as ET
import os
import tensorflow.compat.v1 as tf

NAME_LABEL_MAP = {'back_ground': 0,
                   '焊点': 1,
                   '标签': 2 }

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
  """Opens all TFRecord shards for writing and adds them to an exit stack.

  Args:
    exit_stack: A context2.ExitStack used to automatically closed the TFRecords
      opened in this function.
    base_path: The base path for all shards
    num_shards: The number of shards

  Returns:
    The list of opened TFRecords. Position k in the list corresponds to shard k.
  """
  tf_record_output_filenames = [
      '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
      for idx in range(num_shards)
  ]

  tfrecords = [
      exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
      for file_name in tf_record_output_filenames
  ]
  return tfrecords


def read_xml_gtbox_and_label(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = [] 
    for child_of_root in root:
        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)
        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    label = NAME_LABEL_MAP[child_item.text]
                if child_item.tag == 'polygon':
                    tmp_box = []
                    for node in child_item:
                        tmp_box.append(float(node.text))
                    assert label is not None, 'label is none, error'
                    tmp_box.append(label)
                    box_list.append(tmp_box)

    gtbox_label = np.array(box_list, dtype=np.int32)
    xmin=(gtbox_label[:,0:-1:2].min(axis=1)/img_width).tolist()
    xmax=(gtbox_label[:,0:-1:2].max(axis=1)/img_width).tolist()
    ymin=(gtbox_label[:,1:-1:2].min(axis=1)/img_height).tolist()
    ymax=(gtbox_label[:,1:-1:2].max(axis=1)/img_height).tolist()
    category_ids = gtbox_label[:,-1].tolist()
    return img_height, img_width, xmin,ymin,xmax,ymax,category_ids

def load_txt_annotations(txt_annotation_path):
    with open(txt_annotation_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    return annotations

def read_txt_gtbox_and_label(annotation):
    line = annotation.split()
    image_name = line[0].split('/')[-1]
    bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
    #shape [m,9]
    bboxes = np.reshape(bboxes, [-1, 9])
    x_list   = bboxes[:, 0:-2:2]
    y_list   = bboxes[:, 1:-1:2]
    class_id = (bboxes[:, -1]+1).tolist()
    print(class_id)
    y_max =  (np.max(y_list, axis=1)/2048).tolist()
    y_min =  (np.min(y_list, axis=1)/2048).tolist()
    x_max =  (np.max(x_list, axis=1)/2448).tolist()
    x_min =  (np.min(x_list, axis=1)/2448).tolist()
    print(y_max)
    return image_name,x_min,y_min,x_max,y_max,class_id


def create_tf_example(img_height, img_width,
                      box_xmin,box_ymin,box_xmax,box_ymax,category_ids,
                      image_path):
  img_full_path = image_path
  with tf.gfile.GFile(img_full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  if img_height and img_width:
     image_height = img_height
     image_width = img_width
  else:
    with tf.Session() as sess:
      image = tf.image.decode_png(encoded_jpg)
      shape_tuple=image.eval().shape
      image_height=shape_tuple[0]
      image_width =shape_tuple[1]
  feature_dict = {
      'image/height'    :int64_feature(image_height),
      'image/width'     :int64_feature(image_width),
      'image/encoded'   :bytes_feature(encoded_jpg),
      'image/format'    :bytes_feature('png'.encode('utf8')),}
  xmin=box_xmin
  xmax=box_xmax
  ymin=box_ymin
  ymax=box_ymax
  category_ids=category_ids
  feature_dict.update({
        'image/object/bbox/xmin'  :float_list_feature(xmin),
        'image/object/bbox/xmax'  :float_list_feature(xmax),
        'image/object/bbox/ymin'  :float_list_feature(ymin),
        'image/object/bbox/ymax'  :float_list_feature(ymax),
        'image/object/class/label':int64_list_feature(category_ids)
    })
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example

def create_tf_record_from_xml(image_path,xml_path,tf_output_path,
                              tf_record_num_shards,img_format):

  logging.info('writing to output path: %s', tf_output_path)
  writers = [tf.python_io.TFRecordWriter(tf_output_path+ '-%05d-of-%05d.tfrecord' % (i, tf_record_num_shards))
      for i in range(tf_record_num_shards)]
  for count, xml in enumerate(glob.glob(xml_path + '/*.xml')):
        # to avoid path error in different development platform
        xml = xml.replace('\\', '/')
        img_name = xml.split('/')[-1].split('.')[0] + img_format
        img_path = image_path + '/' + img_name
        if not os.path.exists(img_path):
          print('{} is not exist!'.format(img_path))
          continue
        img_height,img_width,xmin,ymin,xmax,ymax,category_ids=read_xml_gtbox_and_label(xml)
        example = create_tf_example(None, None, xmin, ymin, xmax, ymax, category_ids, img_path)
        #example=create_tf_example(img_height, img_width,xmin,ymin,xmax,ymax,category_ids,img_path)
        writers[count % tf_record_num_shards].write(example.SerializeToString())

def create_tf_record_from_txt(image_dir_path,txt_path,tf_output_path,
                              tf_record_num_shards):

  logging.info('writing to output path: %s', tf_output_path)
  writers = [tf.python_io.TFRecordWriter(tf_output_path+ '-%05d-of-%05d.tfrecord' % (i, tf_record_num_shards))
      for i in range(tf_record_num_shards)]
  annotations=load_txt_annotations(txt_path)
  for count, annotation in enumerate(annotations):
        #to avoid path error in different development platform
        print("****************************")
        image_name,xmin,ymin,xmax,ymax,category_ids=read_txt_gtbox_and_label(annotation)
        img_path = image_dir_path + '/' + image_name
        if not os.path.exists(img_path):
          print('{} is not exist!'.format(img_path))
          continue
        example = create_tf_example(None, None, xmin, ymin, xmax, ymax, category_ids, img_path)
        writers[count % tf_record_num_shards].write(example.SerializeToString())

def main(_):
    #create_tf_record_from_xml(image_path="/home/lwp/efficientdet/dataset/img",
                              #xml_path  ="/home/lwp/efficientdet/dataset/xml",
                              #tf_output_path="/home/lwp/efficientdet/dataset/tfrecord/",
                              #tf_record_num_shards=5,img_format=".png")
    create_tf_record_from_txt(image_dir_path="/home/lwp/tensorflow-yolov3/data/dataset/imgnew",
                              txt_path="/home/lwp/tensorflow-yolov3/data/dataset/trainnew.txt",
                              tf_output_path="/home/lwp/anaconda3/envs/tf/automl/efficientdet/dataset/tfrecord/",
                              tf_record_num_shards=5)

if __name__ == '__main__':
  tf.app.run(main)
