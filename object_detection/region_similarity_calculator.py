from abc import ABCMeta
from abc import abstractmethod
import tensorflow.compat.v1 as tf

def area(boxlist, scope=None):
  with tf.name_scope(scope, 'Area'):
    #shape=[N,1]
    y_min, x_min, y_max, x_max = tf.split(value=boxlist.get(), num_or_size_splits=4, axis=1)
    #shape=[N,1]
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

def intersection(boxlist1, boxlist2, scope=None):
  with tf.name_scope(scope, 'Intersection'):
    #shape=[N,1]
    y_min1, x_min1, y_max1, x_max1 = tf.split(value=boxlist1.get(), num_or_size_splits=4, axis=1)
    #shape=[M,1]
    y_min2, x_min2, y_max2, x_max2 = tf.split(value=boxlist2.get(), num_or_size_splits=4, axis=1)
    #shape=[N,M]
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    #shape=[N,M]
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    # shape=[N,M]
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    # shape=[N,M]
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    # shape=[N,M]
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    # shape=[N,M]
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    # shape=[N,M]
    return intersect_heights * intersect_widths


def iou(boxlist1, boxlist2, scope=None):
  with tf.name_scope(scope, 'IOU'):
    #shape=[N,M]
    intersections = intersection(boxlist1, boxlist2)
    #shape=[N,1]
    areas1 = area(boxlist1)
    #shape=[M,1]
    areas2 = area(boxlist2)
    # shape=[N,M]
    unions = (tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(tf.equal(intersections, 0.0),tf.zeros_like(intersections), tf.truediv(intersections, unions))

class RegionSimilarityCalculator(object):

  __metaclass__ = ABCMeta
  def compare(self, boxlist1, boxlist2, scope=None):
    with tf.name_scope(scope, 'Compare', [boxlist1, boxlist2]) as scope:
      return self._compare(boxlist1, boxlist2)
  @abstractmethod
  def _compare(self, boxlist1, boxlist2):
    pass

class IouSimilarity(RegionSimilarityCalculator):
  def _compare(self, boxlist1, boxlist2):
    return iou(boxlist1, boxlist2)
