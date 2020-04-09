from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import os
from absl import logging
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
from typing import Text, Dict, Any, List
import anchors
import dataloader
import det_model_fn
import hparams_config
import utils
from visualize import vis_utils
coco_id_mapping = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush',
}
weld_id_mapping = {1: 'weld', 2: 'label'}
"""The maximum number of (anchor,class) pairs to keep for non-max suppression"""
MAX_DETECTION_POINTS = 5000
"""
函数名称:image_preprocess(image, image_size: int)
函数功能:归一化原图到满足输入层数据格式
输入参数:
   ---image:输入图像tensor(python环境下numpy的array就是tensor),shape=[height,weight,3]
   ---image_size:输入图像归一化到输入层指定输入格式的尺寸
返回:
  --image      :归一化的图像,shape=[image_size,image_size,3]
  --image_scale:原图缩放比例,type is float32
"""
def image_preprocess(image, image_size: int):
  input_processor = dataloader.DetectionInputProcessor(image, image_size)
  input_processor.normalize_image()
  input_processor.set_scale_factors_to_output_size()
  image = input_processor.resize_and_crop_image()
  image_scale = input_processor.image_scale_to_original
  return image, image_scale

"""
函数名称:build_inputs(image_dir_path: Text, image_size: int)
函数功能:将batch_size大小的测试推理图像统一转换为网络输入层要求输入的格式
输入参数:
     ---image_dir_path : 推理图像保存的文件夹
     ---image_size     : 网络输入层要求输入图像张量的大小
返回:
    --- raw_images:原始图像列表,shape =[batch,height,weight,3]
    --- tf.stack(images):输入层张量,shape =[batch,image_size,image_size,3]
    ----tf.stack(scales):原始图缩放比例张量,shape=[batch,1]
"""
def build_inputs(image_dir_path: Text, image_size: int):
  raw_images, images, scales = [], [], []
  for file_name in os.listdir(image_dir_path):
    f=os.path.join(image_dir_path, file_name)
    image = Image.open(f)
    raw_images.append(image)
    image, scale = image_preprocess(image, image_size)
    images.append(image)
    scales.append(scale)
  return raw_images, tf.stack(images), tf.stack(scales)
"""
函数名称:build_model(model_name: Text, inputs: tf.Tensor, **kwargs)
函数功能:根据指定模型名称(efficientdet-d0/efficientdet-d1)构建网络结构
输入参数:
    ---model_name:模型名称
    ---inputs    :网络输入层输入张量
"""
def build_model(model_name: Text, inputs: tf.Tensor, **kwargs):
  model_arch =   det_model_fn.get_model_arch(model_name)
  class_outputs, box_outputs = model_arch(inputs, model_name, **kwargs)
  return class_outputs, box_outputs

"""
函数名称:restore_ckpt(sess, ckpt_path, enable_ema=True, export_ckpt=None)
函数功能:加载训练ckpt模型
输入参数:
    --sess       :tf_session会话
    --ckpt_path  :ckpt 模型保存的文件夹
    --export_ckpt:在训练构建图基础+后处理处理部分重新保存ckpt模型的文件夹路径
"""
def restore_ckpt(sess, ckpt_path, enable_ema=True, export_ckpt=None):
  sess.run(tf.global_variables_initializer())
  if tf.io.gfile.isdir(ckpt_path):
    ckpt_path = tf.train.latest_checkpoint(ckpt_path)
  if enable_ema:
    ema = tf.train.ExponentialMovingAverage(decay=0.0)
    ema_vars = utils.get_ema_vars()
    var_dict = ema.variables_to_restore(ema_vars)
    ema_assign_op = ema.apply(ema_vars)
  else:
    var_dict = utils.get_ema_vars()
    ema_assign_op = None
  tf.train.get_or_create_global_step()
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(var_dict, max_to_keep=1)
  saver.restore(sess, ckpt_path)
  if export_ckpt:
    print('export model to {}'.format(export_ckpt))
    if ema_assign_op is not None:
      sess.run(ema_assign_op)
    saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    saver.save(sess, export_ckpt)
"""
函数名称:add_metric_fn_inputs(params, cls_outputs, box_outputs, metric_fn_inputs)
函数功能:根据box的类别分数,保留分数最高制定大小的预测结果
输入参数:
    ---params     : 与网络配置有关的配置参数,比如多尺度特征图的min_level,max_level
                    批量训练/推理的batch_size,训练数据集/num_classes
    ---cls_outputs: 多尺度特征图级别和类别预测映射字典,
                    cls_outputs[level] of shape is [batch,feat_size,feat_size,per_anchor_num*class_num]
    ---box_outputs: 多尺度特征图级别和box预测映射字典,
                    box_outputs[level] of shape is [batch,feat_size,feat_size,per_anchor_num*4]          

"""
def add_metric_fn_inputs(params, cls_outputs, box_outputs, metric_fn_inputs):
    cls_outputs_all = []
    box_outputs_all = []
    # Concatenates class and box of all levels into one tensor.
    for level in range(params['min_level'], params['max_level'] + 1):
        cls_outputs_all.append(tf.reshape(cls_outputs[level], [params['batch_size'], -1, params['num_classes']]))
        box_outputs_all.append(tf.reshape(box_outputs[level], [params['batch_size'], -1, 4]))
    # shape=[batch,all_anchor_num,num_class]
    cls_outputs_all = tf.concat(cls_outputs_all, 1)
    #shape=[batch,all_anchor_num,4]
    box_outputs_all = tf.concat(box_outputs_all, 1)
    cls_outputs_all_after_topk = []
    box_outputs_all_after_topk = []
    indices_all = []
    classes_all = []
    for index in range(params['batch_size']):
        #shape=[all_anchor_num,num_class]
        cls_outputs_per_sample = cls_outputs_all[index]
        #shape=[all_anchor_num,4]
        box_outputs_per_sample = box_outputs_all[index]
        #shape=[all_anchor_num*num_class,]
        cls_outputs_per_sample_reshape = tf.reshape(cls_outputs_per_sample, [-1])
        #cls_topk_indices of shape =[MAX_DETECTION_POINTS,]
        _, cls_topk_indices = tf.nn.top_k(cls_outputs_per_sample_reshape, k=MAX_DETECTION_POINTS)
        # Gets top-k class and box scores.
        #保留的box所在的index,shape=[MAX_DETECTION_POINTS,]
        indices = tf.div(cls_topk_indices, params['num_classes'])
        #保留的box对应的class—idx shape=[MAX_DETECTION_POINTS,]
        classes = tf.mod(cls_topk_indices, params['num_classes'])
        #保留的box具有的类别分数/置信度分数(没有归一化)的index shape=[MAX_DETECTION_POINTS,2]
        cls_indices = tf.stack([indices, classes], axis=1)
        # box对应的类别score,shape is [MAX_DETECTION_POINTS,]
        cls_outputs_after_topk = tf.gather_nd(cls_outputs_per_sample, cls_indices)
        cls_outputs_all_after_topk.append(cls_outputs_after_topk)
        #shape is [MAX_DETECTION_POINTS,4]
        box_outputs_after_topk = tf.gather_nd(box_outputs_per_sample, tf.expand_dims(indices, 1))
        box_outputs_all_after_topk.append(box_outputs_after_topk)
        indices_all.append(indices)
        classes_all.append(classes)

    #Concatenates via the batch dimension.
    # shape is [batch_size,MAX_DETECTION_POINTS,]---->score
    cls_outputs_all_after_topk = tf.stack(cls_outputs_all_after_topk, axis=0)
    #shape is [batch_size,MAX_DETECTION_POINTS,4]---ty,tx,th,tw
    box_outputs_all_after_topk = tf.stack(box_outputs_all_after_topk, axis=0)
    #shape is [batch_size,MAX_DETECTION_POINTS]---->indices
    #用于解码时候提取anchor
    indices_all = tf.stack(indices_all, axis=0)
    #shape is [batch_size,MAX_DETECTION_POINTS]---->class_idx
    classes_all = tf.stack(classes_all, axis=0)
    metric_fn_inputs['cls_outputs_all'] = cls_outputs_all_after_topk
    metric_fn_inputs['box_outputs_all'] = box_outputs_all_after_topk
    metric_fn_inputs['indices_all'] = indices_all
    metric_fn_inputs['classes_all'] = classes_all

"""
函数名称:det_post_process(params: Dict[Any, Any], cls_outputs: Dict[int, tf.Tensor],
                         box_outputs: Dict[int, tf.Tensor], scales: List[float])
函数功能:Post preprocessing the box/class predictions
输入参数:
    --params: a parameter dictionary that includes `min_level`, `max_level`,
             `batch_size`, and `num_classes`.
    --cls_outputs: an OrderDict with keys representing levels and values
                   representing logits in [batch_size, height, width, num_anchors].
    --box_outputs: an OrderDict with keys representing levels and values
                   representing box regression targets in [batch_size, height, width, num_anchors * 4].
    --scales: a list of float values indicating image scale. 
  Returns:
    detections_batch: a batch of detection results. Each detection is a tensor
      with each row representing [image_id, x, y, width, height, score, class].         
"""
def det_post_process(params: Dict[Any, Any], cls_outputs: Dict[int, tf.Tensor],
                     box_outputs: Dict[int, tf.Tensor], scales: List[float]):

  outputs = {'cls_outputs_all': [None], 'box_outputs_all': [None],
             'indices_all':     [None], 'classes_all':     [None]}

  add_metric_fn_inputs( params, cls_outputs, box_outputs, outputs)
  #Create anchor_label for picking top-k predictions.
  eval_anchors = anchors.Anchors(params['min_level'],
                                 params['max_level'],
                                 params['num_scales'],
                                 params['aspect_ratios'],
                                 params['anchor_scale'],
                                 params['image_size'])
  anchor_labeler = anchors.AnchorLabeler(eval_anchors, params['num_classes'])
  #Add all detections for each input image.
  detections_batch = []
  for index in range(params['batch_size']):
    #shape is [MAX_DETECTION_POINTS,]---->score
    cls_outputs_per_sample = outputs['cls_outputs_all'][index]
    #shape is [MAX_DETECTION_POINTS,4]---->box ---ty,tx,th,tw
    box_outputs_per_sample = outputs['box_outputs_all'][index]
    # shape is [MAX_DETECTION_POINTS,]
    indices_per_sample = outputs['indices_all'][index]
    # shape is [MAX_DETECTION_POINTS,]
    classes_per_sample = outputs['classes_all'][index]
    detections = anchor_labeler.generate_detections(
        cls_outputs_per_sample, box_outputs_per_sample, indices_per_sample,
        classes_per_sample, image_id=[index], image_scale=[scales[index]],
        disable_pyfun=False)
    detections_batch.append(detections)
  #shape is batch =[batch,M,7]---[image_id, x, y, width, height, score, class]
  return tf.stack(detections_batch, name='detections')

"""
  Visualizes a given image.
  Args:
    image: a image with shape [H, W, C].
    boxes: a box prediction with shape       [N, 4] ordered [ymin, xmin, ymax, xmax].
    classes: a class prediction with shape   [N].
    scores: A list of float value with shape [N].
    id_mapping: a dictionary from class id to name.
    min_score_thresh: minimal score for showing.
                      If claass probability is below this threshold,
                      then the object will not show up.
    max_boxes_to_draw: maximum bounding box to draw.
    line_thickness: how thick is the bounding box line.
    **kwargs: extra parameters.
  Returns:
    output_image: an output image with annotated boxes and classes.
"""
def visualize_image(image,boxes,classes,scores,
                    id_mapping,min_score_thresh=0.2,
                    max_boxes_to_draw=50,line_thickness=4,**kwargs):

  category_index = {k: {'id': k, 'name': id_mapping[k]} for k in id_mapping}
  img = np.array(image)
  vis_utils.visualize_boxes_and_labels_on_image_array(img,boxes,classes,scores,
                                                      category_index,min_score_thresh=min_score_thresh,
                                                      max_boxes_to_draw=max_boxes_to_draw,
                                                      line_thickness=line_thickness,**kwargs)
  return img

#推理引擎----用于batch_size推理/测试图像
class InferenceDriver(object):
  def __init__(self, model_name: Text, ckpt_path: Text, image_size: int = None,
               label_id_mapping: Dict[int, Text] = None):
    self.model_name = model_name
    self.ckpt_path = ckpt_path
    self.label_id_mapping = label_id_mapping or weld_id_mapping
    self.params = hparams_config.get_detection_config(self.model_name).as_dict()
    self.params.update(dict(is_training_bn=False, use_bfloat16=False))
    if image_size:
      self.params.update(dict(image_size=image_size))

  def inference(self,image_path_pattern: Text,output_dir: Text,**kwargs):
    params = copy.deepcopy(self.params)
    with tf.Session() as sess:
      #Buid inputs and preprocessing.
      #raw_images of shape is [batch_size,height,weight,3]
      #images of shape     is [batch_size,image_size,image_size,3]
      #scales of shape     is [batch_size,1]
      raw_images, images, scales = build_inputs(image_path_pattern, params['image_size'])
      #Build model.
      class_outputs, box_outputs = build_model(self.model_name, images, **self.params)
      #加载预训练模型
      restore_ckpt(sess, self.ckpt_path, enable_ema=True, export_ckpt=None)
      #for postprocessing.
      params.update(dict(batch_size=len(raw_images), disable_pyfun=False))
      #Build postprocessing.
      #shape is [batch_size,N,7]
      detections_batch = det_post_process(params, class_outputs, box_outputs, scales)
      outputs_np = sess.run(detections_batch)
      print(outputs_np)
      #Visualize results.
      for i, output_np in enumerate(outputs_np):
        #output_np has format [image_id, x, y, width,height,score, class]
        boxes   = output_np[:, 1:5]
        classes = output_np[:, 6].astype(int)
        scores  = output_np[:, 5]
        #convert [x, y, width, height] to [ymin, xmin, ymax, xmax]
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
        boxes[:, 2:4] += boxes[:, 0:2]
        img = visualize_image(raw_images[i], boxes, classes, scores, self.label_id_mapping, **kwargs)
        output_image_path = os.path.join(output_dir, str(i) + '.jpg')
        Image.fromarray(img).save(output_image_path)
        logging.info('writing file to %s', output_image_path)
    return outputs_np

class ServingDriver(object):
  def __init__(self, model_name: Text,ckpt_path: Text,
               image_size: int = None, batch_size: int = 1,
               label_id_mapping: Dict[int, Text] = None):

    self.model_name =  model_name #模型名称
    self.ckpt_path  =  ckpt_path  #训练ckpt模型保存的路径
    self.batch_size =  batch_size #推理图像的大小---->常见的就是单张推理--batch_size=1
    self.label_id_mapping = label_id_mapping or coco_id_mapping #class_id--->class_name of dict
    #根据模型名称获取对应的默认配置参数
    self.params = hparams_config.get_detection_config(self.model_name).as_dict()
    #更新参数--->train阶段用TPU下的batch_norm,推理下用tf.batch_norm
    self.params.update(dict(is_training_bn=False, use_bfloat16=False))
    #如果用户制定的推理图像大小,则更新默认参数中image_size
    if image_size:
      self.params.update(dict(image_size=image_size))
    #保存计算图的若干输入节点和若干输出节点
    self.signitures = None
    #保存tf_session会话
    self.sess = None

  def build(self):
    """Build model and restore checkpoints."""
    params = copy.deepcopy(self.params)
    image_files = tf.placeholder(tf.string, name='image_files', shape=(None))
    image_size = params['image_size']
    raw_images = []
    for i in range(self.batch_size):
      image = tf.io.decode_image(image_files[i])
      image.set_shape([None, None, None])
      raw_images.append(image)
    raw_images =tf.stack(raw_images, name='image_arrays')
    scales, images =[], []

    #获取满足推理图像数据格式
    for i in range(self.batch_size):
      image, scale = image_preprocess(raw_images[i], image_size)
      scales.append(scale)
      images.append(image)

    #shape=[batch,1]---->用于后处理
    scales = tf.stack(scales)
    #shape=[batch,image_size,image_size,3]---->用于推理
    images = tf.stack(images)
    #构建模型网络结构
    class_outputs, box_outputs = build_model(self.model_name, images, **params)
    params.update(dict(batch_size=self.batch_size, disable_pyfun=False))
    #shape is [batch,M,7]----[image_id, x, y, width, height, score, class]
    detections = det_post_process(params, class_outputs, box_outputs, scales)

    #创建tf_session
    if not self.sess:
      self.sess = tf.Session()
    #load train of ckpt model
    restore_ckpt(self.sess, self.ckpt_path, enable_ema=True, export_ckpt=None)
    print(image_files.name[:-2])
    print(raw_images.name [:-2])
    print(detections.name [:-2])
    self.signitures = {
        'image_files':  image_files,
        'image_arrays': raw_images,
        'prediction':   detections,
    }
    return self.signitures

  def save_inference_pb_model_from_string_img(self,output_dir):
      output_node_names = [self.signitures['image_files'].name[:-2],
                           self.signitures['prediction'].name[:-2]]
      print(output_node_names)
      converted_graph_def = tf.graph_util.convert_variables_to_constants(self.sess,
                             input_graph_def=self.sess.graph.as_graph_def(),
                             output_node_names=output_node_names)
      pb_file = output_dir + "/" + "efficientder-d0.pb"
      # 保存为pb模型
      with tf.gfile.GFile(pb_file, "wb") as f:
        f.write(converted_graph_def.SerializeToString())
      #ress.run()---->输出list[0] of shape [batch_size,m,7]

  def save_inference_pb_model_from_raw_image(self,output_dir):
      output_node_names = [self.signitures['image_arrays'].name[:-2],
                           self.signitures['prediction'].name[:-2]]
      print(output_node_names)
      converted_graph_def = tf.graph_util.convert_variables_to_constants(self.sess,
                                                                         input_graph_def=self.sess.graph.as_graph_def(),
                                                                         output_node_names=output_node_names)
      pb_file = output_dir + "/" + "efficientder-d0-raw-image.pb"
      # 保存为pb模型
      with tf.gfile.GFile(pb_file, "wb") as f:
          f.write(converted_graph_def.SerializeToString())

  def visualize(self, image, predictions, **kwargs):
    """
    Visualize predictions on image.
    Args:
      image: Image content in shape of [height, width, 3].
      predictions: a list of vector, with each vector has the format of
        [image_id, x, y, width, height, score, class].
      **kwargs: extra parameters for for vistualization, such as
        min_score_thresh, max_boxes_to_draw, and line_thickness.

    Returns:
      annotated image.
    """
    boxes   =   predictions[:, 1:5]
    classes =   predictions[:, 6].astype(int)
    scores  =   predictions[:, 5]
    # This is not needed if disable_pyfun=True
    # convert [x, y, width, height] to [ymin, xmin, ymax, xmax]
    boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
    boxes[:, 2:4] += boxes[:, 0:2]
    return visualize_image(image, boxes, classes, scores, self.label_id_mapping,**kwargs)

  """
  函数名称: serve_files(self, image_arrays)
  函数功能: Serve a list of input image files..
  输入参数:
     image_files: a list of image files with shape [1] and type string.
   Returns:
      A list of detections.
  """
  def serve_files(self, image_files: List[Text]):
    if not self.sess:
      self.build()
    predictions = self.sess.run(self.signitures['prediction'],
                  feed_dict={self.signitures['image_files']: image_files})
    return predictions

  """
  函数名称: serve_images(self, image_arrays)
  函数功能: Serve a list of image arrays.
  输入参数:
      --- image_arrays: A list of image content with each image has shape
                        [height, width, 3] and uint8 type.
  Returns:
      shape=[batch,m,7],A list of detections.
  """
  def serve_images(self, image_arrays):
    if not self.sess:
      self.build()
    predictions = self.sess.run(
        self.signitures['prediction'],
        feed_dict={self.signitures['image_arrays']: image_arrays})
    return predictions

  def export(self, output_dir):
    signitures = self.signitures
    signature_def_map = {
        'serving_default':
            tf.saved_model.predict_signature_def(
                {signitures['image_arrays'].name: signitures['image_arrays']},
                {signitures['prediction'].name: signitures['prediction']}),
        'serving_base64':
            tf.saved_model.predict_signature_def(
                {signitures['image_files'].name: signitures['image_files']},
                {signitures['prediction'].name: signitures['prediction']}),
    }

    b = tf.saved_model.Builder(output_dir)
    b.add_meta_graph_and_variables(
        self.sess,
        tags=['serve'],
        signature_def_map=signature_def_map,
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
        clear_devices=True)
    b.save()
    logging.info('Model saved at %s', output_dir)

