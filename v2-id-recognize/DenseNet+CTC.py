# encoding:utf8

import tensorflow as tf
from textGenerator import TextGenerator, get_batches_from_file

textGen = TextGenerator(1)
# image_width = textGen.width
image_height = textGen.height
input_channel = textGen.channel

blocks_num = 3
layers_per_block = 8  # 共3个block，每个block有8层，所以一个block过后，通道数会增加8*8=64个
growth_rate = 12  # 指定一个block中每个层的输出特征图数量
keep_prob = 0.8
class_num = len(textGen.chars) + 1  # 6860
save_path = "traindModel/model"

INITIAL_LN_RATE = 0.0001  # 学习率
DECAY_STEPS = 50000
DECAY_FACTOR = 0.99  # The learning rate decay factor

image_width = tf.placeholder(tf.int32)
inputs = tf.placeholder(tf.float32, [None, image_height, None, input_channel])  # 指定输入张量
targets = tf.sparse_placeholder(tf.int32)
seq_len = tf.placeholder(tf.int32, [None])
is_training = tf.placeholder(tf.bool)  # 指定是否在训练

w_alpha = 0.01
b_alpha = 0.1
# 第一层卷积
w = tf.Variable(w_alpha * tf.random_normal([5, 5, 1, 64]), name='init_conv')
output = tf.nn.conv2d(inputs, w, strides=[1, 2, 2, 1], padding='SAME')  # 这里只有卷积，没有加偏置
width = tf.ceil(image_width / 2)
# 输出[batch_size, 16, 140, 64]

# Dense block
for i in range(blocks_num):
    # 1 实现dense block
    layer_input = output
    for j in range(layers_per_block):
        output = tf.contrib.layers.batch_norm(layer_input, decay=0.9, center=True, scale=True, epsilon=1.1e-5,
                                              is_training=is_training, updates_collections=None, fused=True,
                                              scope='bn1_block%d_layer%d' % (i, j))
        output = tf.nn.relu(output)
        w = tf.Variable(w_alpha * tf.random_normal([3, 3, output.shape.as_list()[-1], growth_rate]),
                        name='w1_block%d_layer%d' % (i, j))
        output = tf.nn.conv2d(output, w, strides=[1, 1, 1, 1], padding='SAME')
        width = tf.ceil(width / 1)
        output = tf.cond(is_training, lambda: tf.nn.dropout(output, keep_prob=keep_prob), lambda: output)
        layer_input = tf.concat([layer_input, output], axis=-1)  # 更新每一层的输入张量
        # 第一个dense block， 通道数变化：64+64=128

    # 2 实现transition layer
    if i != blocks_num - 1:
        output = tf.contrib.layers.batch_norm(layer_input, decay=0.9, center=True, scale=True, epsilon=1.1e-5,
                                              is_training=is_training, updates_collections=None, fused=True,
                                              scope='bn2_block%d_layer%d' % (i, j))
        output = tf.nn.relu(output)
        w = tf.Variable(w_alpha * tf.random_normal([1, 1, output.shape.as_list()[-1], 128]),
                        name='w2_block%d_layer%d' % (i, j))
        output = tf.nn.conv2d(output, w, strides=[1, 1, 1, 1], padding='SAME')
        width = tf.ceil(width / 1)
        output = tf.cond(is_training, lambda: tf.nn.dropout(output, keep_prob=keep_prob), lambda: output)
        output = tf.nn.avg_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        width = tf.ceil((width - 2 + 1) / 2)

output = tf.contrib.layers.batch_norm(layer_input, decay=0.9, center=True, scale=True, epsilon=1.1e-5,
                                      is_training=is_training, updates_collections=None, fused=True, scope='bn3')
output = tf.nn.relu(output)
output = tf.transpose(output, perm=[0, 2, 1, 3])  # 转置矩阵的height和width两个维度,[batch, width, height, channels]
lst = output.get_shape().as_list()
output = tf.reshape(output,
                    [-1, lst[2] * lst[3]])  # [batch*width, height*channels]  注意这里不能是lst[0]*lst[1]，因为他俩不确定，不能计算它们

w_fc1 = tf.Variable(w_alpha * tf.random_normal([lst[2] * lst[3], 1000]))
b_fc1 = tf.Variable(b_alpha * tf.random_normal([1000]))
u_fc1 = tf.nn.bias_add(tf.matmul(output, w_fc1), b_fc1)
a_fc1 = tf.nn.relu(u_fc1)
a_fc1 = tf.cond(is_training, lambda: tf.nn.dropout(a_fc1, keep_prob=keep_prob), lambda: a_fc1)

w_out = tf.Variable(w_alpha * tf.random_normal([1000, class_num]))
b_out = tf.Variable(b_alpha * tf.random_normal([class_num]))
u_out = tf.nn.bias_add(tf.matmul(a_fc1, w_out), b_out)
width = tf.cast(width, tf.int32)  # 因为ceil函数返回值是浮点型的，所以必须先转化为整型
u_out = tf.reshape(u_out,
                   [-1, width, class_num])  # [batch, max_time, num_classes]，注意这里不能是lst[0]和lst[1]，因为他俩不确定，不能计算它们
u_out = tf.transpose(u_out, (1, 0, 2))  # [max_time, batch, num_classes]

loss = tf.nn.ctc_loss(labels=targets, inputs=u_out, sequence_length=seq_len * width)
cost = tf.reduce_mean(loss)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(INITIAL_LN_RATE, global_step, DECAY_STEPS, DECAY_FACTOR, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

# 计算精确度
decoded, log_prob = tf.nn.ctc_beam_search_decoder(u_out, seq_len * width, merge_repeated=False)
distance = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=6)
    try:
        saver.restore(sess, save_path)
        print("model has been loaded successfully!")
    except ValueError:
        print("model does not exist!")

    img, sparse_targets, seq_len1 = get_batches_from_file()
    dec, dis = sess.run([decoded, distance],
                        feed_dict={inputs: img, targets: sparse_targets, seq_len: seq_len1, is_training: False,
                                   image_width: img.shape[-2]})
    print("期望输出：{}".format(sparse_targets[1]))
    print("实际输出：{}".format(dec[0].values))
    print("距离：{}".format(dis))

    # i = 0
    # while True:
    #     img, sparse_targets, seq_len1 = textGen.get_next_batch(30)
    #     # print(img.shape[-2])  280
    #     for j in range(10):
    #         opt, dis = sess.run([optimizer, distance],
    #                             feed_dict={inputs: img, targets: sparse_targets, seq_len: seq_len1, is_training: True,
    #                                        image_width: img.shape[-2]})
    #     print("第{}次：{}".format(i, dis))
    #     if i % 100 == 0 and i != 0:
    #         img, sparse_targets, seq_len1 = textGen.get_next_batch(30)
    #         dec, dis = sess.run([decoded, distance],
    #                             feed_dict={inputs: img, targets: sparse_targets, seq_len: seq_len1, is_training: False,
    #                                        image_width: img.shape[-2]})
    #         print("期望输出：{}".format(sparse_targets[1]))
    #         print("实际输出：{}".format(dec[0].values))
    #         print("距离：{}".format(dis))
    #         saver.save(sess=sess, save_path=save_path, global_step=i)
    #     i += 1
