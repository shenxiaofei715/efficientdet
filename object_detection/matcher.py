from abc import ABCMeta
from abc import abstractmethod
import tensorflow.compat.v1 as tf
class Match(object):
  def __init__(self, match_results):
    if match_results.shape.ndims != 1:
      raise ValueError('match_results should have rank 1')
    if match_results.dtype != tf.int32:
      raise ValueError('match_results should be an int32 or int64 scalar tensor')
    self._match_results = match_results

  @property
  def match_results(self):
    return self._match_results

  #Extension function
  def matched_column_indices(self):
    return self._reshape_and_cast(tf.where(tf.greater(self._match_results, -1)))

  # Extension function
  def matched_column_indicator(self):
    return tf.greater_equal(self._match_results, 0)

  # Extension function
  def num_matched_columns(self):
    return tf.size(self.matched_column_indices())

  # Extension function
  def unmatched_column_indices(self):
    return self._reshape_and_cast(tf.where(tf.equal(self._match_results, -1)))

  # Extension function
  def unmatched_column_indicator(self):
    return tf.equal(self._match_results, -1)

  # Extension function
  def num_unmatched_columns(self):
    return tf.size(self.unmatched_column_indices())

  # Extension function
  def ignored_column_indices(self):
    return self._reshape_and_cast(tf.where(self.ignored_column_indicator()))

  # Extension function
  def ignored_column_indicator(self):
    return tf.equal(self._match_results, -2)

  # Extension function
  def num_ignored_columns(self):
    return tf.size(self.ignored_column_indices())

  # Extension function
  def unmatched_or_ignored_column_indices(self):
    return self._reshape_and_cast(tf.where(tf.greater(0, self._match_results)))

  # Extension function
  def matched_row_indices(self):
    return self._reshape_and_cast(tf.gather(self._match_results, self.matched_column_indices()))

  def _reshape_and_cast(self, t):
    return tf.cast(tf.reshape(t, [-1]), tf.int32)

  def gather_based_on_match(self, input_tensor, unmatched_value,ignored_value):
    input_tensor = tf.concat([tf.stack([ignored_value, unmatched_value]),input_tensor], axis=0)
    gather_indices = tf.maximum(self.match_results + 2, 0)
    gathered_tensor = tf.gather(input_tensor, gather_indices)
    return gathered_tensor


class Matcher(object):
  __metaclass__ = ABCMeta

  def match(self, similarity_matrix, scope=None, **params):
    with tf.name_scope(scope, 'Match', [similarity_matrix, params]) as scope:
      return Match(self._match(similarity_matrix, **params))

  @abstractmethod
  def _match(self, similarity_matrix, **params):
    pass
