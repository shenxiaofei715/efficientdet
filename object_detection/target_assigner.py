import tensorflow.compat.v1 as tf
import box_list
import shape_utils
KEYPOINTS_FIELD_NAME = 'keypoints'
class TargetAssigner(object):
  def __init__(self, similarity_calc, matcher, box_coder,
                     negative_class_weight=1.0, unmatched_cls_target=None):
    self._similarity_calc = similarity_calc
    self._matcher = matcher
    self._box_coder = box_coder
    self._negative_class_weight = negative_class_weight
    if unmatched_cls_target is None:
      self._unmatched_cls_target = tf.constant([0], tf.float32)
    else:
      self._unmatched_cls_target = unmatched_cls_target
  @property
  def box_coder(self):
    return self._box_coder

  def assign(self, anchors, groundtruth_boxes, groundtruth_labels=None,
             groundtruth_weights=None, **params):
    if not isinstance(anchors, box_list.BoxList):
      raise ValueError('anchors must be an BoxList')

    if not isinstance(groundtruth_boxes, box_list.BoxList):
      raise ValueError('groundtruth_boxes must be an BoxList')

    if groundtruth_labels is None:
      #shape=[N,1]
      groundtruth_labels = tf.ones(tf.expand_dims(groundtruth_boxes.num_boxes(),0))
      groundtruth_labels = tf.expand_dims(groundtruth_labels, -1)

    unmatched_shape_assert = shape_utils.assert_shape_equal(
        shape_utils.combined_static_and_dynamic_shape(groundtruth_labels)[1:],
        shape_utils.combined_static_and_dynamic_shape(self._unmatched_cls_target))

    labels_and_box_shapes_assert = shape_utils.assert_shape_equal(
        shape_utils.combined_static_and_dynamic_shape(groundtruth_labels)[:1],
        shape_utils.combined_static_and_dynamic_shape(groundtruth_boxes.get())[:1])

    if groundtruth_weights is None:
      num_gt_boxes = groundtruth_boxes.num_boxes_static()
      if not num_gt_boxes:
        num_gt_boxes = groundtruth_boxes.num_boxes()
      groundtruth_weights = tf.ones([num_gt_boxes], dtype=tf.float32)

    with tf.control_dependencies(
        [unmatched_shape_assert, labels_and_box_shapes_assert]):
      match_quality_matrix = self._similarity_calc.compare(groundtruth_boxes,anchors)
      match = self._matcher.match(match_quality_matrix, **params)
      #shape=[anchor_num,4]
      reg_targets = self._create_regression_targets(anchors,groundtruth_boxes,match)
      #shape=[anchor_num,1]
      cls_targets = self._create_classification_targets(groundtruth_labels,match)
      #shape=[anchor_num,]
      reg_weights = self._create_regression_weights(match, groundtruth_weights)
      #shape=[anchor_num,]
      cls_weights = self._create_classification_weights(match,groundtruth_weights)

    num_anchors = anchors.num_boxes_static()
    if num_anchors is not None:
      reg_targets = self._reset_target_shape(reg_targets, num_anchors)
      cls_targets = self._reset_target_shape(cls_targets, num_anchors)
      reg_weights = self._reset_target_shape(reg_weights, num_anchors)
      cls_weights = self._reset_target_shape(cls_weights, num_anchors)
    return cls_targets, cls_weights, reg_targets, reg_weights, match

  def _create_regression_targets(self, anchors, groundtruth_boxes, match):
    matched_gt_boxes = match.gather_based_on_match(groundtruth_boxes.get(),
                                                   unmatched_value=tf.zeros(4),
                                                   ignored_value=tf.zeros(4))
    matched_gt_boxlist = box_list.BoxList(matched_gt_boxes)
    ####
    if groundtruth_boxes.has_field(KEYPOINTS_FIELD_NAME):
      groundtruth_keypoints = groundtruth_boxes.get_field(KEYPOINTS_FIELD_NAME)
      matched_keypoints = match.gather_based_on_match(
        groundtruth_keypoints,
          unmatched_value=tf.zeros(groundtruth_keypoints.get_shape()[1:]),
          ignored_value=tf.zeros(groundtruth_keypoints.get_shape()[1:]))
      matched_gt_boxlist.add_field(KEYPOINTS_FIELD_NAME, matched_keypoints)
    ####
    matched_reg_targets = self._box_coder.encode(matched_gt_boxlist, anchors)
    match_results_shape = shape_utils.combined_static_and_dynamic_shape(match.match_results)
    unmatched_ignored_reg_targets = tf.tile(self._default_regression_target(), [match_results_shape[0], 1])
    matched_anchors_mask = match.matched_column_indicator()
    reg_targets = tf.where(matched_anchors_mask,matched_reg_targets,unmatched_ignored_reg_targets)
    return reg_targets

  def _default_regression_target(self):
    return tf.constant([self._box_coder.code_size*[0]], tf.float32)

  def _create_classification_targets(self, groundtruth_labels, match):
    return match.gather_based_on_match(
        groundtruth_labels,
        unmatched_value=self._unmatched_cls_target,
        ignored_value=self._unmatched_cls_target)

  def _create_angle_regression_target(self):
      pass

  def _create_angle_class__target(self):
      pass

  def _create_regression_weights(self, match, groundtruth_weights):
    return match.gather_based_on_match(
        groundtruth_weights, ignored_value=0., unmatched_value=0.)

  def _create_classification_weights(self,
                                     match,
                                     groundtruth_weights):
    return match.gather_based_on_match(groundtruth_weights,ignored_value=0.,
                                       unmatched_value=self._negative_class_weight)

  def _reset_target_shape(self, target, num_anchors):
    target_shape = target.get_shape().as_list()
    target_shape[0] = num_anchors
    target.set_shape(target_shape)
    return target

  def get_box_coder(self):
    return self._box_coder
