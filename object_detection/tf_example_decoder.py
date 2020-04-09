import tensorflow.compat.v1 as tf
class TfExampleDecoder(object):
    def __init__(self, include_mask=False, regenerate_source_id=False):
        self._keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string),
            'image/height': tf.FixedLenFeature((), tf.int64, -1),
            'image/width': tf.FixedLenFeature((), tf.int64, -1),
            'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
            'image/object/class/label': tf.VarLenFeature(tf.int64)
        }
    def _decode_image(self, parsed_tensors):
        image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
        image.set_shape([None, None, 3])
        return image

    def _decode_boxes(self, parsed_tensors):
        xmin = parsed_tensors['image/object/bbox/xmin']
        xmax = parsed_tensors['image/object/bbox/xmax']
        ymin = parsed_tensors['image/object/bbox/ymin']
        ymax = parsed_tensors['image/object/bbox/ymax']
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    def _decode_areas(self, parsed_tensors):
        xmin = parsed_tensors['image/object/bbox/xmin']
        xmax = parsed_tensors['image/object/bbox/xmax']
        ymin = parsed_tensors['image/object/bbox/ymin']
        ymax = parsed_tensors['image/object/bbox/ymax']
        area = (xmax - xmin) * (ymax - ymin)
        return area

    def decode(self, serialized_example):
        parsed_tensors = tf.io.parse_single_example(serialized_example, self._keys_to_features)
        for k in parsed_tensors:
            if isinstance(parsed_tensors[k], tf.SparseTensor):
                if parsed_tensors[k].dtype == tf.string:
                    parsed_tensors[k] = tf.sparse_tensor_to_dense(
                        parsed_tensors[k], default_value='')
                else:
                    parsed_tensors[k] = tf.sparse_tensor_to_dense(
                        parsed_tensors[k], default_value=0)

        image = self._decode_image(parsed_tensors)  # 解析图像
        boxes = self._decode_boxes(parsed_tensors)
        areas = self._decode_areas(parsed_tensors)

        decode_image_shape = tf.logical_or(tf.equal(parsed_tensors['image/height'], -1),
                                           tf.equal(parsed_tensors['image/width'], -1))
        image_shape = tf.cast(tf.shape(image), dtype=tf.int64)
        parsed_tensors['image/height'] = tf.where(decode_image_shape, image_shape[0], parsed_tensors['image/height'])
        parsed_tensors['image/width'] = tf.where(decode_image_shape, image_shape[1], parsed_tensors['image/width'])
        decoded_tensors = {
            'image': image,
            'height': parsed_tensors['image/height'],
            'width': parsed_tensors['image/width'],
            'groundtruth_classes': parsed_tensors['image/object/class/label'],
            'groundtruth_area': areas,
            'groundtruth_boxes': boxes,
        }
        return decoded_tensors

class TfExampleDecoderRBox(object):
    def __init__(self):
        self._keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string),
            'image/height': tf.FixedLenFeature((), tf.int64, -1),
            'image/width': tf.FixedLenFeature((), tf.int64, -1),
            'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/angle': tf.VarLenFeature(tf.float32),
            'image/object/class/label': tf.VarLenFeature(tf.int64),
            'image/object/angle-class/label': tf.VarLenFeature(tf.int64),

        }
    def _decode_image(self, parsed_tensors):
        image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
        image.set_shape([None, None, 3])
        return image

    def _decode_rboxes(self, parsed_tensors):
        xmin = parsed_tensors['image/object/bbox/xmin']
        xmax = parsed_tensors['image/object/bbox/xmax']
        ymin = parsed_tensors['image/object/bbox/ymin']
        ymax = parsed_tensors['image/object/bbox/ymax']
        angle = parsed_tensors['image/object/bbox/angle']
        return tf.stack([ymin, xmin, ymax, xmax,angle], axis=-1)

    def _decode_angle_class(self,parsed_tensors):
        return  parsed_tensors['image/object/angle-class/label']

    def _decode_areas(self, parsed_tensors):
        xmin = parsed_tensors['image/object/bbox/xmin']
        xmax = parsed_tensors['image/object/bbox/xmax']
        ymin = parsed_tensors['image/object/bbox/ymin']
        ymax = parsed_tensors['image/object/bbox/ymax']
        area = (xmax - xmin) * (ymax - ymin)
        return area

    def decode(self, serialized_example):
        parsed_tensors = tf.io.parse_single_example(serialized_example, self._keys_to_features)
        for k in parsed_tensors:
            if isinstance(parsed_tensors[k], tf.SparseTensor):
                if parsed_tensors[k].dtype == tf.string:
                    parsed_tensors[k] = tf.sparse_tensor_to_dense(
                        parsed_tensors[k], default_value='')
                else:
                    parsed_tensors[k] = tf.sparse_tensor_to_dense(
                        parsed_tensors[k], default_value=0)

        image = self._decode_image(parsed_tensors)
        boxes = self._decode_rboxes(parsed_tensors)
        areas = self._decode_areas(parsed_tensors)
        angle_class=self._decode_angle_class(parsed_tensors)
        decode_image_shape = tf.logical_or(tf.equal(parsed_tensors['image/height'], -1),
                                           tf.equal(parsed_tensors['image/width'], -1))
        image_shape = tf.cast(tf.shape(image), dtype=tf.int64)
        parsed_tensors['image/height'] = tf.where(decode_image_shape, image_shape[0], parsed_tensors['image/height'])
        parsed_tensors['image/width'] = tf.where(decode_image_shape, image_shape[1], parsed_tensors['image/width'])
        decoded_tensors = {
            'image': image,
            'height': parsed_tensors['image/height'],
            'width': parsed_tensors['image/width'],
            'groundtruth_classes': parsed_tensors['image/object/class/label'],
            'groundtruth_area': areas,
            'groundtruth_boxes': boxes,
            'groundtruth_angle_class': angle_class,
        }
        return decoded_tensors