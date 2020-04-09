from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
import tensorflow.compat.v1 as tf
import argmax_matcher
import box_list
import faster_rcnn_box_coder
import region_similarity_calculator
import target_assigner
# The minimum score to consider a logit for identifying detections.
MIN_CLASS_SCORE = -5.0
# The score for a dummy detection
_DUMMY_DETECTION_SCORE = -1e5
# The maximum number of (anchor,class) pairs to keep for non-max suppression.
MAX_DETECTION_POINTS = 5000
#The maximum number of detections per image.
MAX_DETECTIONS_PER_IMAGE = 100

def decode_box_outputs_tf(rel_codes, anchors):
  ycenter_a = (anchors[0] + anchors[2]) / 2
  xcenter_a = (anchors[1] + anchors[3]) / 2
  ha = anchors[2] - anchors[0]
  wa = anchors[3] - anchors[1]
  ty, tx, th, tw = tf.unstack(rel_codes, num=4)
  w = tf.math.exp(tw) * wa
  h = tf.math.exp(th) * ha
  ycenter = ty * ha + ycenter_a
  xcenter = tx * wa + xcenter_a
  ymin = ycenter - h / 2.
  xmin = xcenter - w / 2.
  ymax = ycenter + h / 2.
  xmax = xcenter + w / 2.
  return tf.stack([ymin, xmin, ymax, xmax], axis=1)

def _generate_detections_tf(cls_outputs, box_outputs, anchor_boxes, indices,
                            classes, image_id, image_scale, num_classes,
                            use_native_nms=False):
  anchor_boxes = tf.gather(anchor_boxes, indices)
  scores = tf.math.sigmoid(cls_outputs)
  boxes = decode_box_outputs_tf(tf.transpose(box_outputs, [1, 0]),
                                tf.transpose(anchor_boxes, [1, 0]))
  def _else(detections, class_id):
    boxes_cls = tf.gather(boxes, indices)
    scores_cls = tf.gather(scores, indices)
    all_detections_cls = tf.concat([tf.reshape(boxes_cls, [-1, 4]), scores_cls],axis=1)
    top_detection_idx = tf.image.non_max_suppression( all_detections_cls[:, :4],
                                                      all_detections_cls[:, 4],
                                                      MAX_DETECTIONS_PER_IMAGE,
                                                      iou_threshold=0.5)
    top_detections_cls = tf.gather(all_detections_cls, top_detection_idx)
    height = top_detections_cls[:, 2] - top_detections_cls[:, 0]
    width = top_detections_cls[:, 3] - top_detections_cls[:, 1]
    top_detections_cls = tf.stack([top_detections_cls[:, 1]* image_scale,
                                   top_detections_cls[:, 0]* image_scale ,
                                   width* image_scale,
                                   height* image_scale,
                                   top_detections_cls[:, 4]], axis=-1)
    top_detections_cls = tf.stack(
        [ tf.cast(tf.tile(image_id, [tf.size(top_detection_idx)]), tf.float32),
          *tf.unstack(top_detections_cls, 5, axis=1),
          tf.cast(tf.tile(tf.expand_dims(class_id + 1.0,axis=0), [tf.size(top_detection_idx)]), tf.float32)],
        axis=1)
    detections = tf.concat([detections, top_detections_cls], axis=0)
    return detections
  detections = tf.constant([], tf.float32, [0, 7])
  for c in range(num_classes):
    indices = tf.where(tf.equal(classes, c))

    detections = tf.cond(tf.equal(tf.shape(indices)[0], 0),
                         lambda: detections,
                         lambda class_id=c: _else(detections, class_id))

  return tf.identity(detections, name='detection')



def _generate_anchor_configs(min_level, max_level, num_scales, aspect_ratios):
  anchor_configs = {}
  for level in range(min_level, max_level + 1):
    anchor_configs[level] = []
    for scale_octave in range(num_scales):
      for aspect in aspect_ratios:
        anchor_configs[level].append( (2**level, scale_octave / float(num_scales), aspect))
  return anchor_configs

def _generate_anchor_boxes(image_size, anchor_scale, anchor_configs):
  boxes_all = []
  for _, configs in anchor_configs.items():
    boxes_level = []
    for config in configs:
      stride, octave_scale, aspect = config
      if image_size % stride != 0:
        raise ValueError('input size must be divided by the stride.')
      base_anchor_size = anchor_scale * stride * 2**octave_scale
      anchor_size_x_2 = base_anchor_size * aspect[0] / 2.0
      anchor_size_y_2 = base_anchor_size * aspect[1] / 2.0

      x = np.arange(stride / 2, image_size, stride)
      y = np.arange(stride / 2, image_size, stride)
      xv, yv = np.meshgrid(x, y)
      xv = xv.reshape(-1)
      yv = yv.reshape(-1)

      boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                         yv + anchor_size_y_2, xv + anchor_size_x_2))
      boxes = np.swapaxes(boxes, 0, 1)
      boxes_level.append(np.expand_dims(boxes, axis=1))
    # concat anchors on the same level to the reshape NxAx4
    boxes_level = np.concatenate(boxes_level, axis=1)
    boxes_all.append(boxes_level.reshape([-1, 4]))

  anchor_boxes = np.vstack(boxes_all)
  return anchor_boxes

class Anchors(object):
  """RetinaNet Anchors class."""

  def __init__(self, min_level, max_level, num_scales, aspect_ratios,
               anchor_scale, image_size):
    self.min_level = min_level
    self.max_level = max_level
    self.num_scales = num_scales
    self.aspect_ratios = aspect_ratios
    self.anchor_scale = anchor_scale
    self.image_size = image_size
    self.config = self._generate_configs()
    self.boxes = self._generate_boxes()
  def _generate_configs(self):
    """Generate configurations of anchor boxes."""
    return _generate_anchor_configs(self.min_level, self.max_level,
                                    self.num_scales, self.aspect_ratios)

  def _generate_boxes(self):
    """Generates multiscale anchor boxes."""
    boxes = _generate_anchor_boxes(self.image_size, self.anchor_scale,
                                   self.config)
    boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
    return boxes

  def get_anchors_per_location(self):
    return self.num_scales * len(self.aspect_ratios)

class AnchorLabeler(object):
  """Labeler for multiscale anchor boxes."""

  def __init__(self, anchors, num_classes, match_threshold=0.5):
    similarity_calc = region_similarity_calculator.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(match_threshold,
                                           unmatched_threshold=match_threshold,
                                           negatives_lower_than_unmatched=True,
                                           force_match_for_each_row=True)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()
    self._target_assigner = target_assigner.TargetAssigner(similarity_calc, matcher, box_coder)
    self._anchors = anchors
    self._match_threshold = match_threshold
    self._num_classes = num_classes
  def _unpack_labels(self, labels):
    """Unpacks an array of labels into multiscales labels."""
    labels_unpacked = collections.OrderedDict()
    anchors = self._anchors
    count = 0
    for level in range(anchors.min_level, anchors.max_level + 1):
      feat_size = int(anchors.image_size / 2**level)
      steps = feat_size**2 * anchors.get_anchors_per_location()
      indices = tf.range(count, count + steps)
      count += steps
      labels_unpacked[level] = tf.reshape(tf.gather(labels, indices), [feat_size, feat_size, -1])
    return labels_unpacked

  def label_anchors(self, gt_boxes, gt_labels):
    gt_box_list = box_list.BoxList(gt_boxes)
    anchor_box_list = box_list.BoxList(self._anchors.boxes)
    #cls_weights, box_weights are not used
    cls_targets, _, box_targets, _, matches = self._target_assigner.assign(
        anchor_box_list, gt_box_list, gt_labels)
    # class labels start from 1 and the background class = -1
    cls_targets -= 1
    cls_targets = tf.cast(cls_targets, tf.int32)
    # Unpack labels.
    cls_targets_dict = self._unpack_labels(cls_targets)
    box_targets_dict = self._unpack_labels(box_targets)
    num_positives = tf.reduce_sum(tf.cast(tf.not_equal(matches.match_results, -1), tf.float32))
    return cls_targets_dict, box_targets_dict, num_positives

  def generate_detections(self, cls_outputs, box_outputs, indices, classes,
                          image_id, image_scale, disable_pyfun=None):
    return _generate_detections_tf(cls_outputs, box_outputs,
                                   self._anchors.boxes, indices, classes,
                                    image_id, image_scale, self._num_classes)

