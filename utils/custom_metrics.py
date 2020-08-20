import tensorflow as tf


class EuclideanDistanceMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the euclidian distance
    """

    def __init__(self, **kwargs):
        if 'is_training' in kwargs:
            if kwargs['is_training']:
                super(EuclideanDistanceMetric, self).__init__(name='Euclidean Distance')
            else:
                super(EuclideanDistanceMetric, self).__init__()
        else:
            super(EuclideanDistanceMetric, self).__init__()
        self.l2_norms = self.add_weight("norm2", initializer="zeros")
        self.count = self.add_weight("counter", initializer="zeros")

    # def __init__(self, **kwargs):
    #     super(EuclideanDistanceMetric, self).__init__( **kwargs)
    #     self.l2_norms = self.add_weight("norm2", initializer="zeros")
    #     self.count = self.add_weight("counter", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        self.l2_norms.assign_add(tf.norm(y_pred - y_true, ord='euclidean'))
        self.count.assign_add(1)
        # tf.print('update_state ', self.l2_norms, ' ', self.count)

    def result(self) -> tf.Tensor:
        val = self.l2_norms / self.count
        # tf.print('result ', self.l2_norms, ' ', self.count, ' ', val)
        return val

    def reset_states(self):
        # tf.print('reset_states ', self.l2_norms)
        self.l2_norms.assign(0)
        self.count.assign(0)



class AUCMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the euclidian distance
    """

    def __init__(self, **kwargs):
        if 'is_training' in kwargs:
            if kwargs['is_training']:
                super(EuclideanDistanceMetric, self).__init__(name='AUC')
            else:
                super(EuclideanDistanceMetric, self).__init__()
        else:
            super(EuclideanDistanceMetric, self).__init__()

        self.l2_norms = self.add_weight("norm2", initializer="zeros")
        self.count = self.add_weight("counter", initializer="zeros")

    # def __init__(self, **kwargs):
    #     super(EuclideanDistanceMetric, self).__init__( **kwargs)
    #     self.l2_norms = self.add_weight("norm2", initializer="zeros")
    #     self.count = self.add_weight("counter", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        self.l2_norms.assign_add(tf.norm(y_pred - y_true, ord='euclidean'))
        self.count.assign_add(1)
        # tf.print('update_state ', self.l2_norms, ' ', self.count)

    def result(self) -> tf.Tensor:
        val = self.l2_norms / self.count
        # tf.print('result ', self.l2_norms, ' ', self.count, ' ', val)
        return val

    def reset_states(self):
        # tf.print('reset_states ', self.l2_norms)
        self.l2_norms.assign(0)
        self.count.assign(0)
