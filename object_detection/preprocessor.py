import tensorflow.compat.v1 as tf
import box_list

def _flip_boxes_left_right(boxes):
  ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
  flipped_xmin = tf.subtract(1.0, xmax)
  flipped_xmax = tf.subtract(1.0, xmin)
  flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)
  return flipped_boxes

def _flip_masks_left_right(masks):
  return masks[:, :, ::-1]

def keypoint_flip_horizontal(keypoints, flip_point, flip_permutation,scope=None):
  with tf.name_scope(scope, 'FlipHorizontal'):
    keypoints = tf.transpose(keypoints, [1, 0, 2])
    keypoints = tf.gather(keypoints, flip_permutation)
    v, u = tf.split(value=keypoints, num_or_size_splits=2, axis=2)
    u = flip_point * 2.0 - u
    new_keypoints = tf.concat([v, u], 2)
    new_keypoints = tf.transpose(new_keypoints, [1, 0, 2])
    return new_keypoints

def random_horizontal_flip(image,
                           boxes=None,
                           masks=None,
                           keypoints=None,
                           keypoint_flip_permutation=None,
                           seed=None):
  def _flip_image(image):
    image_flipped = tf.image.flip_left_right(image)
    return image_flipped

  if keypoints is not None and keypoint_flip_permutation is None:
    raise ValueError('keypoints are provided but keypoints_flip_permutation is not provided')

  with tf.name_scope('RandomHorizontalFlip', values=[image, boxes]):
    result = []
    # random variable defining whether to do flip or not
    do_a_flip_random = tf.greater(tf.random_uniform([], seed=seed), 0.5)
    # flip image
    image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)
    result.append(image)
    # flip boxes
    if boxes is not None:
      boxes = tf.cond(do_a_flip_random, lambda: _flip_boxes_left_right(boxes),
                      lambda: boxes)
      result.append(boxes)
    # flip masks
    if masks is not None:
      masks = tf.cond(do_a_flip_random, lambda: _flip_masks_left_right(masks),lambda: masks)
      result.append(masks)

    # flip keypoints
    if keypoints is not None and keypoint_flip_permutation is not None:
      permutation = keypoint_flip_permutation
      keypoints = tf.cond(
          do_a_flip_random,
          lambda: keypoint_flip_horizontal(keypoints, 0.5, permutation),
          lambda: keypoints)
      result.append(keypoints)
    return tuple(result)


def _compute_new_static_size(image, min_dimension, max_dimension):
  """Compute new static shape for resize_to_range method."""
  image_shape = image.get_shape().as_list()
  orig_height = image_shape[0]
  orig_width = image_shape[1]
  num_channels = image_shape[2]
  orig_min_dim = min(orig_height, orig_width)
  # Calculates the larger of the possible sizes
  large_scale_factor = min_dimension / float(orig_min_dim)
  # Scaling orig_(height|width) by large_scale_factor will make the smaller
  # dimension equal to min_dimension, save for floating point rounding errors.
  # For reasonably-sized images, taking the nearest integer will reliably
  # eliminate this error.
  large_height = int(round(orig_height * large_scale_factor))
  large_width = int(round(orig_width * large_scale_factor))
  large_size = [large_height, large_width]
  if max_dimension:
    # Calculates the smaller of the possible sizes, use that if the larger
    # is too big.
    orig_max_dim = max(orig_height, orig_width)
    small_scale_factor = max_dimension / float(orig_max_dim)
    # Scaling orig_(height|width) by small_scale_factor will make the larger
    # dimension equal to max_dimension, save for floating point rounding
    # errors. For reasonably-sized images, taking the nearest integer will
    # reliably eliminate this error.
    small_height = int(round(orig_height * small_scale_factor))
    small_width = int(round(orig_width * small_scale_factor))
    small_size = [small_height, small_width]
    new_size = large_size
    if max(large_size) > max_dimension:
      new_size = small_size
  else:
    new_size = large_size
  return tf.constant(new_size + [num_channels])


def _compute_new_dynamic_size(image, min_dimension, max_dimension):
  """Compute new dynamic shape for resize_to_range method."""
  image_shape = tf.shape(image)
  orig_height = tf.to_float(image_shape[0])
  orig_width = tf.to_float(image_shape[1])
  num_channels = image_shape[2]
  orig_min_dim = tf.minimum(orig_height, orig_width)
  # Calculates the larger of the possible sizes
  min_dimension = tf.constant(min_dimension, dtype=tf.float32)
  large_scale_factor = min_dimension / orig_min_dim
  # Scaling orig_(height|width) by large_scale_factor will make the smaller
  # dimension equal to min_dimension, save for floating point rounding errors.
  # For reasonably-sized images, taking the nearest integer will reliably
  # eliminate this error.
  large_height = tf.to_int32(tf.round(orig_height * large_scale_factor))
  large_width = tf.to_int32(tf.round(orig_width * large_scale_factor))
  large_size = tf.stack([large_height, large_width])
  if max_dimension:
    # Calculates the smaller of the possible sizes, use that if the larger
    # is too big.
    orig_max_dim = tf.maximum(orig_height, orig_width)
    max_dimension = tf.constant(max_dimension, dtype=tf.float32)
    small_scale_factor = max_dimension / orig_max_dim
    # Scaling orig_(height|width) by small_scale_factor will make the larger
    # dimension equal to max_dimension, save for floating point rounding
    # errors. For reasonably-sized images, taking the nearest integer will
    # reliably eliminate this error.
    small_height = tf.to_int32(tf.round(orig_height * small_scale_factor))
    small_width = tf.to_int32(tf.round(orig_width * small_scale_factor))
    small_size = tf.stack([small_height, small_width])
    new_size = tf.cond(
        tf.to_float(tf.reduce_max(large_size)) > max_dimension,
        lambda: small_size, lambda: large_size)
  else:
    new_size = large_size
  return tf.stack(tf.unstack(new_size) + [num_channels])


def resize_to_range(image,
                    masks=None,
                    min_dimension=None,
                    max_dimension=None,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=False,
                    pad_to_max_dimension=False):
  if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  with tf.name_scope('ResizeToRange', values=[image, min_dimension]):
    if image.get_shape().is_fully_defined():
      new_size = _compute_new_static_size(image, min_dimension, max_dimension)
    else:
      new_size = _compute_new_dynamic_size(image, min_dimension, max_dimension)
    new_image = tf.image.resize_images(
        image, new_size[:-1], method=method, align_corners=align_corners)

    if pad_to_max_dimension:
      new_image = tf.image.pad_to_bounding_box(
          new_image, 0, 0, max_dimension, max_dimension)

    result = [new_image]
    if masks is not None:
      new_masks = tf.expand_dims(masks, 3)
      new_masks = tf.image.resize_images(
          new_masks,
          new_size[:-1],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
          align_corners=align_corners)
      new_masks = tf.squeeze(new_masks, 3)
      if pad_to_max_dimension:
        new_masks = tf.image.pad_to_bounding_box(
            new_masks, 0, 0, max_dimension, max_dimension)
      result.append(new_masks)

    result.append(new_size)
    return result


def _copy_extra_fields(boxlist_to_copy_to, boxlist_to_copy_from):
  for field in boxlist_to_copy_from.get_extra_fields():
    boxlist_to_copy_to.add_field(field, boxlist_to_copy_from.get_field(field))
  return boxlist_to_copy_to


def box_list_scale(boxlist, y_scale, x_scale, scope=None):
  with tf.name_scope(scope, 'Scale'):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_boxlist = box_list.BoxList(
        tf.concat([y_min, x_min, y_max, x_max], 1))
    return _copy_extra_fields(scaled_boxlist, boxlist)


def keypoint_scale(keypoints, y_scale, x_scale, scope=None):
  with tf.name_scope(scope, 'Scale'):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    new_keypoints = keypoints * [[[y_scale, x_scale]]]
    return new_keypoints


def scale_boxes_to_pixel_coordinates(image, boxes, keypoints=None):
  boxlist = box_list.BoxList(boxes)
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  scaled_boxes = box_list_scale(boxlist, image_height, image_width).get()
  result = [image, scaled_boxes]
  if keypoints is not None:
    scaled_keypoints = keypoint_scale(keypoints, image_height, image_width)
    result.append(scaled_keypoints)
  return tuple(result)
