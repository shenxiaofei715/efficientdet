from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
import tensorflow.compat.v1 as tf
FASTER_RCNN = 'faster_rcnn'
KEYPOINT = 'keypoint'
MEAN_STDDEV = 'mean_stddev'
SQUARE = 'square'
class BoxCoder(object):
  __metaclass__ = ABCMeta

  @abstractproperty
  def code_size(self):
    pass

  def encode(self, boxes, anchors):
    with tf.name_scope('Encode'):
       return self._encode(boxes, anchors)

  def decode(self, rel_codes, anchors):
    with tf.name_scope('Decode'):
      return self._decode(rel_codes, anchors)

  @abstractmethod
  def _encode(self, boxes, anchors):
    pass

  @abstractmethod
  def _decode(self, rel_codes, anchors):
    pass

def batch_decode(encoded_boxes, box_coder, anchors):
  encoded_boxes.get_shape().assert_has_rank(3)
  if encoded_boxes.get_shape()[1].value != anchors.num_boxes_static():
    raise ValueError('The number of anchors inferred from encoded_boxes'
                     ' and anchors are inconsistent: shape[1] of encoded_boxes'
                     ' %s should be equal to the number of anchors: %s.' %
                     (encoded_boxes.get_shape()[1].value,
                      anchors.num_boxes_static()))

  decoded_boxes =tf.stack([
      box_coder.decode(boxes, anchors).get()
      for boxes in tf.unstack(encoded_boxes)
  ])
  return decoded_boxes
