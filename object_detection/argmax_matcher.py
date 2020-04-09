import tensorflow.compat.v1 as tf
from object_detection import matcher
from object_detection import shape_utils
#matches of shape is [anchor_num,-1]
#below_unmatched  of value is   -1,
#between_thresholds of value is -2
#match of value is ground_box of index
class ArgMaxMatcher(matcher.Matcher):
  def __init__(self,
               matched_threshold,
               unmatched_threshold=None,
               negatives_lower_than_unmatched=True,
               force_match_for_each_row=False):
    if (matched_threshold is None) and (unmatched_threshold is not None):
      raise ValueError('Need to also define matched_threshold when unmatched_threshold is defined')

    self._matched_threshold = matched_threshold
    if unmatched_threshold is None:
      self._unmatched_threshold = matched_threshold
    else:
      if unmatched_threshold > matched_threshold:
        raise ValueError('unmatched_threshold needs to be smaller or equal to matched_threshold')
      self._unmatched_threshold = unmatched_threshold

    if not negatives_lower_than_unmatched:
      if self._unmatched_threshold == self._matched_threshold:
        raise ValueError('When negatives are in between matched and '
                         'unmatched thresholds, these cannot be of equal '
                         'value. matched: %s, unmatched: %s',
                         self._matched_threshold, self._unmatched_threshold)

    self._force_match_for_each_row       = force_match_for_each_row
    self._negatives_lower_than_unmatched = negatives_lower_than_unmatched
  #**params is expander
  def _match(self, similarity_matrix,**params):

    def _match_when_rows_are_empty():
      similarity_matrix_shape = shape_utils.combined_static_and_dynamic_shape(similarity_matrix)
      return -1 * tf.ones([similarity_matrix_shape[1]], dtype=tf.int32)

    def _match_when_rows_are_non_empty():

      matches = tf.argmax(similarity_matrix, 0, output_type=tf.int32)

      if self._matched_threshold is not None:
        matched_vals = tf.reduce_max(similarity_matrix, 0)
        below_unmatched_threshold = tf.greater(self._unmatched_threshold,matched_vals)
        between_thresholds = tf.logical_and(tf.greater_equal(matched_vals, self._unmatched_threshold),
                                            tf.greater(self._matched_threshold, matched_vals))

        if self._negatives_lower_than_unmatched:
          matches = self._set_values_using_indicator(matches,below_unmatched_threshold,-1)
          matches = self._set_values_using_indicator(matches,between_thresholds,-2)
        else:
          matches = self._set_values_using_indicator(matches,below_unmatched_threshold,-2)
          matches = self._set_values_using_indicator(matches,between_thresholds,-1)

      if self._force_match_for_each_row:
        similarity_matrix_shape = shape_utils.combined_static_and_dynamic_shape(similarity_matrix)
        force_match_column_ids = tf.argmax(similarity_matrix, 1,output_type=tf.int32)
        force_match_column_indicators = tf.one_hot(
            force_match_column_ids, depth=similarity_matrix_shape[1])
        force_match_row_ids = tf.argmax(force_match_column_indicators, 0,output_type=tf.int32)
        force_match_column_mask = tf.cast(tf.reduce_max(force_match_column_indicators, 0), tf.bool)
        final_matches = tf.where(force_match_column_mask,force_match_row_ids, matches)
        return final_matches
      else:
        return matches

    if similarity_matrix.shape.is_fully_defined():
      if similarity_matrix.shape[0].value == 0:
        return _match_when_rows_are_empty()
      else:
        return _match_when_rows_are_non_empty()
    else:
      return tf.cond(tf.greater(tf.shape(similarity_matrix)[0], 0),
                     _match_when_rows_are_non_empty,
                     _match_when_rows_are_empty)

  def _set_values_using_indicator(self, x, indicator, val):
    indicator = tf.cast(indicator, x.dtype)
    return tf.add(tf.multiply(x, 1 - indicator), val * indicator)
