from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
from typing import Text, Tuple, List
import det_model_fn
import hparams_config
import inference
import utils
flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model.')
flags.DEFINE_string('logdir', '/tmp/deff/', 'log directory.')
flags.DEFINE_string('runmode', 'infer', 'Run mode: {freeze, bm, dry}')
flags.DEFINE_string('trace_filename', None, 'Trace file name.')
flags.DEFINE_integer('num_classes', 2, 'Number of classes.')
flags.DEFINE_integer('input_image_size', None, 'Size of input image.')
flags.DEFINE_integer('threads', 0, 'Number of threads.')
flags.DEFINE_integer('bm_runs', 20, 'Number of benchmark runs.')
flags.DEFINE_string('tensorrt', None, 'TensorRT mode: {None, FP32, FP16, INT8}')
flags.DEFINE_bool('delete_logdir', True, 'Whether to delete logdir.')
flags.DEFINE_bool('freeze', False, 'Freeze graph.')
flags.DEFINE_bool('xla', False, 'Run with xla optimization.')

flags.DEFINE_string('ckpt_path', "/home/lwp/anaconda3/envs/tf/automl/efficientdet/model/ckpt/efficientdet-d0/weld",'checkpoint dir used for eval.')
flags.DEFINE_string('export_ckpt', None, 'Path for exporting new models.')
flags.DEFINE_bool('enable_ema', True, 'Use ema variables for eval.')

flags.DEFINE_string('input_image', "/home/lwp/anaconda3/envs/tf/automl/efficientdet/testdata/", 'Input image path for inference.')
flags.DEFINE_string('output_image_dir', "/home/lwp/anaconda3/envs/tf/automl/efficientdet/testcocodata/", 'Output dir for inference.')

# For visualization.
flags.DEFINE_integer('line_thickness', None, 'Line thickness for box.')
flags.DEFINE_integer('max_boxes_to_draw', None, 'Max number of boxes to draw.')
flags.DEFINE_float('min_score_thresh', None, 'Score threshold to show box.')

# For saved model.
flags.DEFINE_string('saved_model_dir', '/home/lwp/anaconda3/envs/tf/automl/efficientdet/model/pbtxt/efficientdet-d0',
                    'Folder path for saved model.')

FLAGS = flags.FLAGS


class ModelInspector(object):
  """A simple helper class for inspecting a model."""

  def __init__(self,
               model_name: Text,
               image_size: int,
               num_classes: int,
               logdir: Text,
               tensorrt: Text = False,
               use_xla: bool = False,
               ckpt_path: Text = None,
               enable_ema: bool = True,
               export_ckpt: Text = None,
               saved_model_dir: Text = None):
    self.model_name = model_name
    self.model_params = hparams_config.get_detection_config(model_name)
    self.logdir = logdir
    self.tensorrt = tensorrt
    self.use_xla = use_xla
    self.ckpt_path = ckpt_path
    self.enable_ema = enable_ema
    self.export_ckpt = export_ckpt
    self.saved_model_dir = saved_model_dir

    if image_size:
      # Use user specified image size.
      self.model_overrides = {
          'image_size': image_size,
          'num_classes': num_classes
      }
    else:
      # Use default size.
      image_size = hparams_config.get_detection_config(model_name).image_size
      self.model_overrides = {'num_classes': num_classes}

    # A few fixed parameters.
    batch_size = 1
    self.num_classes = num_classes
    self.inputs_shape = [batch_size, image_size, image_size, 3]
    self.labels_shape = [batch_size, self.num_classes]
    self.image_size = image_size

  def build_model(self, inputs: tf.Tensor,
                  is_training: bool = False) -> List[tf.Tensor]:
    """Build model with inputs and labels and print out model stats."""
    logging.info('start building model')
    model_arch = det_model_fn.get_model_arch(self.model_name)
    cls_outputs, box_outputs = model_arch(
        inputs,
        model_name=self.model_name,
        is_training_bn=is_training,
        use_bfloat16=False,
        **self.model_overrides)

    print('backbone+fpn+box params/flops = {:.6f}M, {:.9f}B'.format(
        *utils.num_params_flops()))

    all_outputs = list(cls_outputs.values()) + list(box_outputs.values())
    return all_outputs

  def inference_image(self, image_image_path, output_dir, **kwargs):
    driver = inference.InferenceDriver(self.model_name, self.ckpt_path,
                                       self.image_size)
    driver.inference(image_image_path, output_dir, **kwargs)

  def export_pb_model_image_string(self):
    driver = inference.ServingDriver(self.model_name, self.ckpt_path)
    driver.build()
    driver.save_inference_pb_model_from_string_img(self.saved_model_dir)

  def export_pb_model_image_arrays(self):
    driver = inference.ServingDriver(self.model_name, self.ckpt_path)
    driver.build()
    driver.save_inference_pb_model_from_raw_image(self.saved_model_dir)

  def read_pb_return_tensors(self,graph, pb_file, return_elements):
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
      frozen_graph_def = tf.GraphDef()
      frozen_graph_def.ParseFromString(f.read())
    with graph.as_default():
       return_elements = tf.import_graph_def(frozen_graph_def,return_elements=return_elements)
    return return_elements

  def restore_pb_model_inference_image_string(self,image_path_pattern,output_dir):
    """Perform inference for the given saved model."""
    return_elements =["image_files:0","detections:0"]
    graph = tf.Graph()
    pb_file=self.saved_model_dir+"/"+"efficientder-d0.pb"
    return_tensors = self.read_pb_return_tensors(graph, pb_file, return_elements)
    with tf.Session(graph=graph) as sess:
      for file_name in os.listdir(image_path_pattern):
        print(os.path.join(image_path_pattern, file_name))
        file_path = os.path.join(image_path_pattern, file_name)
        raw_images = Image.open(file_path)
        raw_data_encode = tf.gfile.FastGFile(file_path, 'rb').read()
        pred_bbox = sess.run([return_tensors[1]],feed_dict={return_tensors[0]:[raw_data_encode]})
        for i, output_np in enumerate(pred_bbox[0]):
          # output_np has format [image_id, x, y, width,  height,score, class]
          boxes = output_np[:, 1:5]
          classes = output_np[:, 6].astype(int)
          scores = output_np[:, 5]
          boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
          boxes[:, 2:4] += boxes[:, 0:2]
          img = inference.visualize_image(
          raw_images, boxes, classes, scores, inference.coco_id_mapping)
          output_image_path = os.path.join(output_dir, "output_"+file_name)
          Image.fromarray(img).save(output_image_path)
          logging.info('writing file to %s', output_image_path)



  def saved_model_inference(self, image_path_pattern, output_dir):
    """Perform inference for the given saved model."""
    with tf.Session() as sess:
      tf.saved_model.load(sess, ['serve'], self.saved_model_dir)
      raw_images = []
      image = Image.open(image_path_pattern)
      raw_images.append(np.array(image))
      outputs_np = sess.run('detections:0', {'image_arrays:0': raw_images})
      for i, output_np in enumerate(outputs_np):
        # output_np has format [image_id, x, y, width,height, score, class]
        boxes = output_np[:, 1:5]
        classes = output_np[:, 6].astype(int)
        scores = output_np[:, 5]
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
        boxes[:, 2:4] += boxes[:, 0:2]
        img = inference.visualize_image(
            raw_images[i], boxes, classes, scores, inference.coco_id_mapping)
        output_image_path = os.path.join(output_dir, str(i) + '.jpg')
        Image.fromarray(img).save(output_image_path)
        logging.info('writing file to %s', output_image_path)

  def build_and_save_model(self):
    """build and save the model into self.logdir."""
    with tf.Graph().as_default(), tf.Session() as sess:
      # Build model with inputs and labels.
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      outputs = self.build_model(inputs, is_training=False)

      # Run the model
      inputs_val = np.random.rand(*self.inputs_shape).astype(float)
      labels_val = np.zeros(self.labels_shape).astype(np.int64)
      labels_val[:, 0] = 1
      sess.run(tf.global_variables_initializer())
      # Run a single train step.
      sess.run(outputs, feed_dict={inputs: inputs_val})
      all_saver = tf.train.Saver(save_relative_paths=True)
      all_saver.save(sess, os.path.join(self.logdir, self.model_name))

      tf_graph = os.path.join(self.logdir, self.model_name + '_train.pb')
      with tf.io.gfile.GFile(tf_graph, 'wb') as f:
        f.write(sess.graph_def.SerializeToString())

  def restore_model(self, sess, ckpt_path, enable_ema=True, export_ckpt=None):
    """Restore variables from a given checkpoint."""
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.latest_checkpoint(ckpt_path)
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
    saver.restore(sess, checkpoint)
    if export_ckpt:
      print('export model to {}'.format(export_ckpt))
      if ema_assign_op is not None:
        sess.run(ema_assign_op)
      saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
      saver.save(sess, export_ckpt)

  def eval_ckpt(self):
    """build and save the model into self.logdir."""
    with tf.Graph().as_default(), tf.Session() as sess:
      # Build model with inputs and labels.
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      self.build_model(inputs, is_training=False)
      self.restore_model(
          sess, self.ckpt_path, self.enable_ema, self.export_ckpt)



  def freeze_model(self) -> Tuple[Text, Text]:
    """Freeze model and convert them into tflite and tf graph."""
    with tf.Graph().as_default(), tf.Session() as sess:
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      outputs = self.build_model(inputs, is_training=False)

      checkpoint = tf.train.latest_checkpoint(self.logdir)
      logging.info('Loading checkpoint: %s', checkpoint)
      saver = tf.train.Saver()

      # Restore the Variables from the checkpoint and freeze the Graph.
      saver.restore(sess, checkpoint)

      output_node_names = [node.name.split(':')[0] for node in outputs]
      graphdef = tf.graph_util.convert_variables_to_constants(
          sess, sess.graph_def, output_node_names)

    return graphdef

  def benchmark_model(self, warmup_runs, bm_runs, num_threads,
                      trace_filename=None):
    """Benchmark model."""
    if self.tensorrt:
      print('Using tensorrt ', self.tensorrt)
      self.build_and_save_model()
      graphdef = self.freeze_model()

    if num_threads > 0:
      print('num_threads for benchmarking: {}'.format(num_threads))
      sess_config = tf.ConfigProto(
          intra_op_parallelism_threads=num_threads,
          inter_op_parallelism_threads=1)
    else:
      sess_config = tf.ConfigProto()

    # rewriter_config_pb2.RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.dependency_optimization = 2
    if self.use_xla:
      sess_config.graph_options.optimizer_options.global_jit_level = (
          tf.OptimizerOptions.ON_2)

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      output = self.build_model(inputs, is_training=False)

      img = np.random.uniform(size=self.inputs_shape)

      sess.run(tf.global_variables_initializer())
      if self.tensorrt:
        fetches = [inputs.name] + [i.name for i in output]
        goutput = self.convert_tr(graphdef, fetches)
        inputs, output = goutput[0], goutput[1:]

      if not self.use_xla:
        # Don't use tf.group because XLA removes the whole graph for tf.group.
        output = tf.group(*output)
      for i in range(warmup_runs):
        start_time = time.time()
        sess.run(output, feed_dict={inputs: img})
        print('Warm up: {} {:.4f}s'.format(i, time.time() - start_time))
      print('Start benchmark runs total={}'.format(bm_runs))
      timev = []
      for i in range(bm_runs):
        if trace_filename and i == (bm_runs // 2):
          run_options = tf.RunOptions()
          run_options.trace_level = tf.RunOptions.FULL_TRACE
          run_metadata = tf.RunMetadata()
          sess.run(output, feed_dict={inputs: img},
                   options=run_options, run_metadata=run_metadata)
          logging.info('Dumping trace to %s', trace_filename)
          trace_dir = os.path.dirname(trace_filename)
          if not tf.io.gfile.exists(trace_dir):
            tf.io.gfile.makedirs(trace_dir)
          with tf.io.gfile.GFile(trace_filename, 'w') as trace_file:
            from tensorflow.python.client import timeline  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            trace_file.write(
                trace.generate_chrome_trace_format(show_memory=True))

        start_time = time.time()
        sess.run(output, feed_dict={inputs: img})
        timev.append(time.time() - start_time)

      timev.sort()
      timev = timev[2:bm_runs-2]
      print('{} {}runs {}threads: mean {:.4f} std {:.4f} min {:.4f} max {:.4f}'
            .format(self.model_name, len(timev), num_threads, np.mean(timev),
                    np.std(timev), np.min(timev), np.max(timev)))

  def convert_tr(self, graph_def, fetches):
    """Convert to TensorRT."""
    from tensorflow.python.compiler.tensorrt import trt  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    converter = trt.TrtGraphConverter(
        nodes_blacklist=[t.split(':')[0] for t in fetches],
        input_graph_def=graph_def,
        precision_mode=self.tensorrt)
    infer_graph = converter.convert()
    goutput = tf.import_graph_def(infer_graph, return_elements=fetches)
    return goutput

  def run_model(self, runmode, threads=0):
    """Run the model on devices."""
    if runmode == 'dry':
      self.build_and_save_model()
    elif runmode == 'freeze':
      self.build_and_save_model()
      self.freeze_model()
    elif runmode == 'ckpt':
      self.eval_ckpt()
    elif runmode == 'infer':
      config_dict = {}
      if FLAGS.line_thickness:
        config_dict['line_thickness'] = FLAGS.line_thickness
      if FLAGS.max_boxes_to_draw:
        config_dict['max_boxes_to_draw'] = FLAGS.max_boxes_to_draw
      if FLAGS.min_score_thresh:
        config_dict['min_score_thresh'] = FLAGS.min_score_thresh
      self.inference_image(FLAGS.input_image, FLAGS.output_image_dir,**config_dict)
    elif runmode == 'saved_model':
      #self.export_saved_model()
      #self.export_pb_model()
      self.restore_pb_model_inference_image_string(FLAGS.input_image,FLAGS.output_image_dir)
    elif runmode == 'saved_model_infer':
      self.saved_model_inference(FLAGS.input_image, FLAGS.output_image_dir)
    elif runmode == 'bm':
      self.benchmark_model(warmup_runs=5, bm_runs=FLAGS.bm_runs,
                           num_threads=threads,
                           trace_filename=FLAGS.trace_filename)


def main(_):
  if tf.io.gfile.exists(FLAGS.logdir) and FLAGS.delete_logdir:
    logging.info('Deleting log dir ...')
    tf.io.gfile.rmtree(FLAGS.logdir)

  inspector = ModelInspector(
      model_name=FLAGS.model_name,
      image_size=FLAGS.input_image_size,
      num_classes=FLAGS.num_classes,
      logdir=FLAGS.logdir,
      tensorrt=FLAGS.tensorrt,
      use_xla=FLAGS.xla,
      ckpt_path=FLAGS.ckpt_path,
      enable_ema=FLAGS.enable_ema,
      export_ckpt=FLAGS.export_ckpt,
      saved_model_dir=FLAGS.saved_model_dir)
  inspector.run_model(FLAGS.runmode, FLAGS.threads)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.disable_v2_behavior()
  tf.app.run(main)
