import tensorflow.compat.v1 as tf
class BoxList(object):
  def __init__(self, boxes):
    if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 4:
      raise ValueError('Invalid dimensions for box data.')
    if boxes.dtype != tf.float32:
      raise ValueError('Invalid tensor type: should be tf.float32')
    self.data = {'boxes': boxes}

  def num_boxes(self):
    return tf.shape(self.data['boxes'])[0]

  def num_boxes_static(self):
    return self.data['boxes'].get_shape()[0].value

  def get_all_fields(self):
    return self.data.keys()

  def get_extra_fields(self):
    return [k for k in self.data.keys() if k != 'boxes']

  def add_field(self, field, field_data):
    self.data[field] = field_data

  def has_field(self, field):
    return field in self.data

  def get(self):
    return self.get_field('boxes')

  def set(self, boxes):
    if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 4:
      raise ValueError('Invalid dimensions for box data.')
    self.data['boxes'] = boxes

  def get_field(self, field):
    if not self.has_field(field):
      raise ValueError('field ' + str(field) + ' does not exist')
    return self.data[field]

  def set_field(self, field, value):
    if not self.has_field(field):
      raise ValueError('field %s does not exist' % field)
    self.data[field] = value

  def get_center_coordinates_and_sizes(self, scope=None):
    with tf.name_scope(scope, 'get_center_coordinates_and_sizes'):
      box_corners = self.get()
      ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(box_corners))
      width = xmax - xmin
      height = ymax - ymin
      ycenter = ymin + height / 2.
      xcenter = xmin + width / 2.
      return [ycenter, xcenter, height, width]

  def transpose_coordinates(self, scope=None):
    with tf.name_scope(scope, 'transpose_coordinates'):
      y_min, x_min, y_max, x_max = tf.split(
          value=self.get(), num_or_size_splits=4, axis=1)
      self.set(tf.concat([x_min, y_min, x_max, y_max], 1))

  def as_tensor_dict(self, fields=None):
    tensor_dict = {}
    if fields is None:
      fields = self.get_all_fields()
    for field in fields:
      if not self.has_field(field):
        raise ValueError('boxlist must contain all specified fields')
      tensor_dict[field] = self.get_field(field)
    return tensor_dict