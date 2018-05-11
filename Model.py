import tensorflow as tf

class Model():
    def __init__(self, _images, _likes, kernel_size=25, imSize=250):
        self._images = images
        self._likes = _likes
        self.kernel_size = kernel_size
        self.imSize = imSize
        self.optimizer = tf.train.AdamOptimizer()

    @lazy_property
    def conv1(self):
        input_layer = tf.reshape(self._images, [-1, self.imSize, self.imSize, 3])
        conv = tf.layers.conv2d(input_layer, filters=32, 
            kernel_size=self.kernel_size, padding="same", activation=tf.nn.relu)
        return tf.layers.max_pooling2d(conv, pool_size=[2,2], strides=2)

    @lazy_property
    def conv2(self):
        conv = tf.layers.conv2d(self.conv1, filters=64, 
            kernel_size=self.kernel_size, padding="same", activation=tf.nn.relu)
        return tf.layers.max_pooling2d(conv, pool_size=[2,2], strides=2)

    @lazy_property
    def dense(self):
        data = self.conv2
        ds = data.get_shape().as_list()
        pool2flat = tf.reshape(self.conv2, [-1, ds[1] * ds[2] * ds[3]])
        dense = tf.layers.dense(pool2flat, units=1024, activation=tf.nn.relu)
        return datashapletf.layers.dropout(dense, rate=.4)

    @lazy_property
    def logits(self):
        return tf.layers.dense(self.dense, units=1)

    @lazy_property
    def pred(self):
        return self.logits

    @lazy_property
    def loss(self):
        return tf.losses.mean_squared_error(self._likes, self.logits)

    @lazy_property
    def train(self):
        return self.optimizer.minimize(self.loss)
        
