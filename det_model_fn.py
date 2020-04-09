from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

import anchors
import efficientdet_arch
import hparams_config
import retinanet_arch
import utils
_DEFAULT_BATCH_SIZE = 64


def update_learning_rate_schedule_parameters(params):
  batch_size = (params['batch_size'] * params['num_shards'] if params['use_tpu']
                else params['batch_size'])
  params['adjusted_learning_rate'] = (params['learning_rate'] * batch_size /
                                      _DEFAULT_BATCH_SIZE)
  steps_per_epoch = params['num_examples_per_epoch'] / batch_size
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
  params['first_lr_drop_step'] = int(params['first_lr_drop_epoch'] *
                                     steps_per_epoch)
  params['second_lr_drop_step'] = int(params['second_lr_drop_epoch'] *
                                      steps_per_epoch)
  params['total_steps'] = int(params['num_epochs'] * steps_per_epoch)


def stepwise_lr_schedule(adjusted_learning_rate, lr_warmup_init,
                         lr_warmup_step, first_lr_drop_step,
                         second_lr_drop_step, global_step):
  logging.info('LR schedule method: stepwise')
  linear_warmup = (lr_warmup_init +
                   (tf.cast(global_step, dtype=tf.float32) / lr_warmup_step *
                    (adjusted_learning_rate - lr_warmup_init)))
  learning_rate = tf.where(global_step < lr_warmup_step,
                           linear_warmup, adjusted_learning_rate)
  lr_schedule = [[1.0, lr_warmup_step],
                 [0.1, first_lr_drop_step],
                 [0.01, second_lr_drop_step]]
  for mult, start_global_step in lr_schedule:
    learning_rate = tf.where(global_step < start_global_step, learning_rate,
                             adjusted_learning_rate * mult)
  return learning_rate

def cosine_lr_schedule(adjusted_lr, lr_warmup_init, lr_warmup_step,
                       total_steps, step):
  logging.info('LR schedule method: cosine')
  linear_warmup = (
      lr_warmup_init +
      (tf.cast(step, dtype=tf.float32) / lr_warmup_step *
       (adjusted_lr - lr_warmup_init)))
  cosine_lr = 0.5 * adjusted_lr * (
      1 + tf.cos(np.pi * tf.cast(step, tf.float32) / total_steps))
  return tf.where(step < lr_warmup_step, linear_warmup, cosine_lr)


def learning_rate_schedule(params, global_step):
  lr_decay_method = params['lr_decay_method']
  if lr_decay_method == 'stepwise':
    return stepwise_lr_schedule(params['adjusted_learning_rate'],
                                params['lr_warmup_init'],
                                params['lr_warmup_step'],
                                params['first_lr_drop_step'],
                                params['second_lr_drop_step'], global_step)
  elif lr_decay_method == 'cosine':
    return cosine_lr_schedule(params['adjusted_learning_rate'],
                              params['lr_warmup_init'],
                              params['lr_warmup_step'],
                              params['total_steps'], global_step)
  else:
    raise ValueError('unknown lr_decay_method: {}'.format(lr_decay_method))


def focal_loss(logits, targets, alpha, gamma, normalizer):
  with tf.name_scope('focal_loss'):
    positive_label_mask = tf.equal(targets, 1.0)
    cross_entropy = (
        tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    # Below are comments/derivations for computing modulator.
    # For brevity, let x = logits,  z = targets, r = gamma, and p_t = sigmod(x)
    # for positive samples and 1 - sigmoid(x) for negative examples.
    #
    # The modulator, defined as (1 - P_t)^r, is a critical part in focal loss
    # computation. For r > 0, it puts more weights on hard examples, and less
    # weights on easier ones. However if it is directly computed as (1 - P_t)^r,
    # its back-propagation is not stable when r < 1. The implementation here
    # resolves the issue.
    #
    # For positive samples (labels being 1),
    #    (1 - p_t)^r
    #  = (1 - sigmoid(x))^r
    #  = (1 - (1 / (1 + exp(-x))))^r
    #  = (exp(-x) / (1 + exp(-x)))^r
    #  = exp(log((exp(-x) / (1 + exp(-x)))^r))
    #  = exp(r * log(exp(-x)) - r * log(1 + exp(-x)))
    #  = exp(- r * x - r * log(1 + exp(-x)))
    #
    # For negative samples (labels being 0),
    #    (1 - p_t)^r
    #  = (sigmoid(x))^r
    #  = (1 / (1 + exp(-x)))^r
    #  = exp(log((1 / (1 + exp(-x)))^r))
    #  = exp(-r * log(1 + exp(-x)))
    #
    # Therefore one unified form for positive (z = 1) and negative (z = 0)
    # samples is:
    #      (1 - p_t)^r = exp(-r * z * x - r * log(1 + exp(-x))).
    neg_logits = -1.0 * logits
    modulator = tf.exp(gamma * targets * neg_logits - gamma * tf.log1p(tf.exp(neg_logits)))
    loss = modulator * cross_entropy
    weighted_loss = tf.where(positive_label_mask, alpha * loss,(1.0 - alpha) * loss)
    weighted_loss /= normalizer
  return weighted_loss


def _classification_loss(cls_outputs,
                         cls_targets,
                         num_positives,
                         alpha=0.25,
                         gamma=2.0):
  normalizer = num_positives
  classification_loss = focal_loss(cls_outputs, cls_targets, alpha, gamma,
                                   normalizer)
  return classification_loss


def _box_loss(box_outputs, box_targets, num_positives, delta=0.1):
  normalizer = num_positives * 4.0
  mask = tf.not_equal(box_targets, 0.0)
  box_loss = tf.losses.huber_loss(
      box_targets,
      box_outputs,
      weights=mask,
      delta=delta,
      reduction=tf.losses.Reduction.SUM)
  box_loss /= normalizer
  return box_loss


def detection_loss(cls_outputs, box_outputs, labels, params):
  num_positives_sum = tf.reduce_sum(labels['mean_num_positives']) + 1.0
  levels = cls_outputs.keys()
  cls_losses = []
  box_losses = []
  for level in levels:
    cls_targets_at_level = tf.one_hot(
        labels['cls_targets_%d' % level],
        params['num_classes'])
    bs, width, height, _, _ = cls_targets_at_level.get_shape().as_list()
    cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                      [bs, width, height, -1])
    box_targets_at_level = labels['box_targets_%d' % level]
    cls_loss = _classification_loss(
        cls_outputs[level],
        cls_targets_at_level,
        num_positives_sum,
        alpha=params['alpha'],
        gamma=params['gamma'])
    cls_loss = tf.reshape(cls_loss,
                          [bs, width, height, -1, params['num_classes']])
    cls_loss *= tf.cast(tf.expand_dims(
        tf.not_equal(labels['cls_targets_%d' % level], -2), -1), tf.float32)
    cls_losses.append(tf.reduce_sum(cls_loss))
    box_losses.append(
        _box_loss(
            box_outputs[level],
            box_targets_at_level,
            num_positives_sum,
            delta=params['delta']))

  # Sum per level losses to total loss.
  cls_loss = tf.add_n(cls_losses)
  box_loss = tf.add_n(box_losses)
  total_loss = cls_loss + params['box_loss_weight'] * box_loss
  return total_loss, cls_loss, box_loss

def reg_l2_loss(weight_decay, regex=r'.*(kernel|weight):0$'):
  """Return regularization l2 loss loss."""
  var_match = re.compile(regex)
  return weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if var_match.match(v.name)])

def _model_fn(features, labels, mode, params, model, variable_filter_fn=None):
  def _model_outputs():
    return model(features, config=hparams_config.Config(params))

  if params['use_bfloat16']:
    with tf.tpu.bfloat16_scope():
      cls_outputs, box_outputs = _model_outputs()
      levels = cls_outputs.keys()
      for level in levels:
        cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
        box_outputs[level] = tf.cast(box_outputs[level], tf.float32)
  else:
    cls_outputs, box_outputs = _model_outputs()
    levels = cls_outputs.keys()

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'image': features,
    }
    for level in levels:
      predictions['cls_outputs_%d' % level] = cls_outputs[level]
      predictions['box_outputs_%d' % level] = box_outputs[level]
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  update_learning_rate_schedule_parameters(params)
  global_step = tf.train.get_or_create_global_step()
  learning_rate = learning_rate_schedule(params, global_step)

  det_loss, cls_loss, box_loss = detection_loss(cls_outputs, box_outputs,labels, params)
  l2loss = reg_l2_loss(params['weight_decay'])
  total_loss = det_loss + l2loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    utils.scalar('lrn_rate', learning_rate)
    utils.scalar('trainloss/cls_loss', cls_loss)
    utils.scalar('trainloss/box_loss', box_loss)
    utils.scalar('trainloss/det_loss', det_loss)
    utils.scalar('trainloss/l2_loss', l2loss)
    utils.scalar('trainloss/loss', total_loss)

  moving_average_decay = params['moving_average_decay']
  if moving_average_decay:
    ema = tf.train.ExponentialMovingAverage(
        decay=moving_average_decay, num_updates=global_step)
    ema_vars = utils.get_ema_vars()

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=params['momentum'])
    if params['use_tpu']:
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    var_list = tf.trainable_variables()
    if variable_filter_fn:
      var_list = variable_filter_fn(var_list, params['resnet_depth'])

    if params.get('clip_gradients_norm', 0) > 0:
      logging.info('clip gradients norm by %f', params['clip_gradients_norm'])
      grads_and_vars = optimizer.compute_gradients(total_loss, var_list)
      with tf.name_scope('clip'):
        grads = [gv[0] for gv in grads_and_vars]
        tvars = [gv[1] for gv in grads_and_vars]
        clipped_grads, gnorm = tf.clip_by_global_norm(
            grads, params['clip_gradients_norm'])
        utils.scalar('gnorm', gnorm)
        grads_and_vars = list(zip(clipped_grads, tvars))

      with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)
    else:
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step, var_list=var_list)

    if moving_average_decay:
      with tf.control_dependencies([train_op]):
        train_op = ema.apply(ema_vars)

  else:
    train_op = None
  eval_metrics = None

  checkpoint = params.get('ckpt') or params.get('backbone_ckpt')
  if checkpoint and mode == tf.estimator.ModeKeys.TRAIN:
    # Initialize the model from an EfficientDet or backbone checkpoint.
    if params.get('ckpt') and params.get('backbone_ckpt'):
      raise RuntimeError('--backbone_ckpt and --checkpoint are mutually exclusive')
    elif params.get('backbone_ckpt'):
      var_scope = params['backbone_name'] + '/'
      if params['ckpt_var_scope'] is None:
        # Use backbone name as default checkpoint scope.
        ckpt_scope = params['backbone_name'] + '/'
      else:
        ckpt_scope = params['ckpt_var_scope'] + '/'
    else:
      # Load every var in the given checkpoint
      var_scope = ckpt_scope = '/'

    def scaffold_fn():
      """Loads pretrained model through scaffold function."""
      logging.info('restore variables from %s', checkpoint)
      var_map = utils.get_ckpt_var_map(
          ckpt_path=checkpoint,
          ckpt_scope=ckpt_scope,
          var_scope=var_scope,
          var_exclude_expr=params.get('var_exclude_expr', None))
      tf.train.init_from_checkpoint(checkpoint, var_map)
      return tf.train.Scaffold()

  elif mode == tf.estimator.ModeKeys.EVAL and moving_average_decay:
    def scaffold_fn():
      """Load moving average variables for eval."""
      logging.info('Load EMA vars with ema_decay=%f', moving_average_decay)
      restore_vars_dict = ema.variables_to_restore(ema_vars)
      saver = tf.train.Saver(restore_vars_dict)
      return tf.train.Scaffold(saver=saver)
  else:
    scaffold_fn = None

  return tf.estimator.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=total_loss,
      train_op=train_op,
      eval_metrics=eval_metrics,
      host_call=utils.get_tpu_host_call(global_step, params),
      scaffold_fn=scaffold_fn)


def retinanet_model_fn(features, labels, mode, params):
  """RetinaNet model."""
  return _model_fn(
      features,
      labels,
      mode,
      params,
      model=retinanet_arch.retinanet,
      variable_filter_fn=retinanet_arch.remove_variables)


def efficientdet_model_fn(features, labels, mode, params):
  """EfficientDet model."""
  return _model_fn(
      features,
      labels,
      mode,
      params,
      model=efficientdet_arch.efficientdet)


def get_model_arch(model_name='efficientdet-d0'):
  """Get model architecture for a given model name."""
  if 'retinanet' in model_name:
    return retinanet_arch.retinanet
  elif 'efficientdet' in model_name:
    return efficientdet_arch.efficientdet
  else:
    raise ValueError('Invalide model name {}'.format(model_name))


def get_model_fn(model_name='efficientdet-d0'):
  """Get model fn for a given model name."""
  if 'retinanet' in model_name:
    return retinanet_model_fn
  elif 'efficientdet' in model_name:
    return efficientdet_model_fn
  else:
    raise ValueError('Invalide model name {}'.format(model_name))
