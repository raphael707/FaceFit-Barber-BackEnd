import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="FaceShape")
class SpatialAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            activation="sigmoid",
            kernel_initializer="glorot_uniform",
            name=self.name + "_conv"
        )

        self.conv.build((input_shape[0], input_shape[1], input_shape[2], 2))
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

        concat = tf.concat([avg_pool, max_pool], axis=-1)

        attention_map = self.conv(concat)

        return inputs * attention_map

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size
        })
        return config


@tf.keras.utils.register_keras_serializable(package="FaceShape")
class TemperatureSoftmax(tf.keras.layers.Layer):
    def __init__(self, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = float(temperature)

    def call(self, logits):
        return tf.nn.softmax(logits / self.temperature, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "temperature": self.temperature
        })
        return config


@tf.keras.utils.register_keras_serializable(package="FaceShape")
class FocalCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=1.5, alpha=1.0, label_smoothing=0.0, **kwargs):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(
            y_pred,
            tf.keras.backend.epsilon(),
            1.0 - tf.keras.backend.epsilon()
        )

        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + (
                self.label_smoothing / num_classes
            )

        cross_entropy = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        true_class_prob = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - true_class_prob, self.gamma)

        return self.alpha * focal_weight * cross_entropy

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing,
        })
        return config