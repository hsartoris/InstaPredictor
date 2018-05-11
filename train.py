import tensorflow as tf
import os
import numpy as np

batchSize = 32
imSize = 250
c1k=25

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

batches_per_epoch = int(len(files)/batchSize)

files = tf.constant(files)
likes = tf.constant(np.expand_dims(np.loadtxt(dataDir + "ids", dtype=int),1))

dataset = tf.data.Dataset.from_tensor_slices((files, likes))
dataset = dataset.map(_parse_function).batch(batchSize)
iterator = dataset.make_initializable_iterator()
iter_init_op = iterator.make_initializer(dataset)

_images, _likes = iterator.get_next()


    
#input_layer = tf.reshape(_images, [-1,imSize,imSize,3])
#conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=c1k,
#    padding="same", activation=tf.nn.relu)
#pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2], strides=2)
#conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=c1k,
#    padding="same", activation=tf.nn.relu)
#pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
#pool2shape = pool2.get_shape().as_list()
#pool2size = pool2shape[1] * pool2shape[2] * pool2shape[3]
#
#pool2_flat = tf.reshape(pool2, [-1, pool2size])
#dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
#dropout = tf.layers.dropout(inputs=dense, rate=.4)
#
#logits = tf.layers.dense(inputs=dropout, units=1)
#print(logits.get_shape().as_list())
#predictions = logits
#print(_likes.get_shape().as_list())
#
#loss = tf.losses.mean_squared_error(_likes, logits)
#optimizer = tf.train.AdamOptimizer()
#train_op = optimizer.minimize(loss)

#report_tensor_allocations_upon_oom
m = Model(_images, _likes)
loss = m.loss
train_op = m.train

    
init = tf.global_variables_initializer()
epochs = 100
with tf.Session() as sess:
    sess.run(init)
    sess.run(iter_init_op)
    for e in range(epochs):
        for eStep in range(batches_per_epoch):
            _, lossval = sess.run([train_op, loss])
            loss = np.sum(lossval)/batchSize
        print("epoch", e,"\tloss:", loss)
