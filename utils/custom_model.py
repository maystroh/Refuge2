import tensorflow as tf


class MySequential(tf.keras.Sequential):

    # def __init__(self, model, num_step_epoch=None, file_writer=None):
    #     super().__init__(model)
    #     self.tb_writer = file_writer
    #     self.steps_per_epoch = num_step_epoch
    #     self.num_steps = 1

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        x, y = data
        # tf.print(y)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                regularization_losses=self.losses,
            )
        # tf.print(y_pred)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)  # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        output = {m.name: m.result() for m in self.metrics}
        # tf.print(output)

        # if self.file_writer is not None:
        #     with self.tb_writer.as_default():
        #         tf.summary.image("Training data", x, step=self.num_steps)

        # self.num_steps = self.num_steps + 1

        return output
