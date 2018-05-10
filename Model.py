import tensorflow as tf
import os
import numpy as np

batchSize = 64
imSize = 500
c1k=50

def _parse_function(fname, likes):
    image_str = tf.read_file(fname)
    image_decoded = tf.image.decode_jpeg(image_str)
    image_resized = tf.image.resize_images(image_decoded, [imSize,imSize])
    return image_resized, likes

dataDir = "images/processed/"

files = []
for fname in os.listdir(dataDir):
    if ".jpg" in fname:
        files.append(dataDir + fname)
files = tf.constant(files)
likes = tf.constant(np.loadtxt(dataDir + "ids", dtype=int))

dataset = tf.data.Dataset.from_tensor_slices((files, likes))
dataset = dataset.map(_parse_function).batch(batchSize)
iterator = dataset.make_initializable_iterator()

_images, _likes = iterator.get_next()


def cnn_model(features, likes, mode):
    
    input_layer = tf.reshape(features, [-1,500,500,3])
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=c1k,
        padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=c1k,
        padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
    pool2shape = pool2.get_shape().as_list()
    pool2size = pool2shape[1] * pool2shape[2] * pool2shape[3]

    pool2_flat = tf.reshape(pool2, [-1, pool2size])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=.4,
        training=mode == tf.estimator.ModeKeys.TRAIN)
    
    logits = tf.layers.dense(inputs=dropout, units=1)
    predictions = logits
    
    loss = tf.losses.mean_squared_error(likes, logits)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    

cnn_model(_images, _likes)
