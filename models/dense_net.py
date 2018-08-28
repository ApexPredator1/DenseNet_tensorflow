import os
import time
import shutil
from datetime import timedelta

import numpy as np
import tensorflow as tf

TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))


class DenseNet:
    def __init__(self, data_provider, growth_rate, depth,
                 total_blocks, keep_prob, num_inter_threads, num_intra_threads,
                 weight_decay, nesterov_momentum, model_type, dataset,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 reduction=1.0,
                 bc_mode=False,
                 **kwargs):
        """
        Class to implement networks from this paper
        https://arxiv.org/pdf/1611.05552.pdf

        Args:
            data_provider: Class, that have all required data sets
            growth_rate: `int`, variable from paper
            depth: `int`, variable from paper
            total_blocks: `int`, paper value == 3
            keep_prob: `float`, keep probability for dropout. If keep_prob = 1
                dropout will be disables
            weight_decay: `float`, weight decay for L2 loss, paper = 1e-4
            nesterov_momentum: `float`, momentum for Nesterov optimizer
            model_type: `str`, 'DenseNet' or 'DenseNet-BC'. Should model use
                bottle neck connections or not.
            dataset: `str`, dataset name
            should_save_logs: `bool`, should logs be saved or not
            should_save_model: `bool`, should model be saved or not
            renew_logs: `bool`, remove previous logs for current model
            reduction: `float`, reduction Theta at transition layer for
                DenseNets with bottleneck layers. See paragraph 'Compression'
                https://arxiv.org/pdf/1608.06993v3.pdf#4
            bc_mode: `bool`, should we use bottleneck layers and features
                reduction or not.
        """
        self.data_provider = data_provider
        self.data_shape = data_provider.data_shape
        self.n_classes = data_provider.n_classes
        self.depth = depth
        self.growth_rate = growth_rate
        self.num_inter_threads = num_inter_threads
        self.num_intra_threads = num_intra_threads
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        self.reduction = reduction
        if not bc_mode:
            print("Build %s model with %d blocks, "
                  "%d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block))
        if bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            print("Build %s model with %d blocks, "
                  "%d bottleneck layers and %d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block,
                      self.layers_per_block))
        print("Reduction at transition layers: %.1f" % self.reduction)

        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        self.dataset_name = dataset
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs
        self.batches_step = 0

        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()

        # Specify the CPU inter and Intra threads used by MKL
        config.intra_op_parallelism_threads = self.num_intra_threads
        config.inter_op_parallelism_threads = self.num_inter_threads

        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf_ver = int(tf.__version__.split('.')[1])
        if TF_VERSION <= 0.10:
            self.sess.run(tf.initialize_all_variables())
            logswriter = tf.train.SummaryWriter
        else:
            self.sess.run(tf.global_variables_initializer())
            logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_writer = logswriter(self.logs_path)

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    @property
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = 'saves/%s' % self.model_identifier
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model.chkpt')
            self._save_path = save_path
        return save_path

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = 'logs/%s' % self.model_identifier
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        return "{}_growth_rate={}_depth={}_dataset_{}".format(
            self.model_type, self.growth_rate, self.depth, self.dataset_name)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception as e:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.save_path)
        self.saver.restore(self.sess, self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    def log_loss_accuracy(self, loss, accuracy, epoch, prefix,
                          should_print=True):
        if should_print:
            print("mean cross_entropy: %f, mean accuracy: %f" % (
                loss, accuracy))
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(
                tag='accuracy_%s' % prefix, simple_value=float(accuracy))
        ])
        self.summary_writer.add_summary(summary, epoch)

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='labels')
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[])

    def _build_graph(self):
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block

        # first - initial 3 x 3 conv to first_output_features
        kernel = tf.get_variable(name='kernel_first',
                                 shape=[3, 3, int(self.images.get_shape()[-1]), self.first_output_features],
                                 initializer=tf.contrib.layers.variance_scaling_initializer())
        output = tf.nn.conv2d(self.images, kernel, strides=[1, 1, 1, 1], padding='SAME')

        # 添加total_blocks个block
        for block in range(self.total_blocks):
            layer_input = output       # layer_input用于更新迭代一个block中每个层的输入，其初始值output是每个block的输出
            # a1层的输入是input
            # a2层的输入是input+a1
            # a3层的输入是input+a1+a2
            # a4层的输入是input+a1+a2+a3
            # ............. 依次迭代
            for layer in range(layers_per_block):
                if not self.bc_mode:    # 没有bottleneck层
                    # 直接BN+ReLU+3*3卷积层
                    output = tf.contrib.layers.batch_norm(layer_input, scale=True, is_training=self.is_training,
                                                          updates_collections=None)
                    output = tf.nn.relu(output)
                    kernel = tf.get_variable(name='kernel_$d_%d' % (block, layer),
                                             shape=[3, 3, int(output.get_shape()[-1]), growth_rate],
                                             initializer=tf.contrib.layers.variance_scaling_initializer())
                    output = tf.nn.conv2d(output, kernel, [1, 1, 1, 1], padding='SAME')
                    output = tf.cond(self.is_training, lambda: tf.nn.dropout(output, self.keep_prob), lambda: output)
                elif self.bc_mode:      # 带有bottleneck层
                    # BN+ReLU+1*1卷积层
                    output = tf.contrib.layers.batch_norm(layer_input, scale=True, is_training=self.is_training,
                                                          updates_collections=None)
                    output = tf.nn.relu(output)
                    kernel = tf.get_variable(name='kernel_$d_%d' % (block, layer),
                                             shape=[1, 1, int(output.get_shape()[-1]), growth_rate * 4],
                                             initializer=tf.contrib.layers.variance_scaling_initializer())
                    output = tf.nn.conv2d(output, kernel, [1, 1, 1, 1], padding='VALID')
                    output = tf.cond(self.is_training, lambda: tf.nn.dropout(output, self.keep_prob), lambda: output)

                    # BN+ReLU+3*3卷积层
                    output = tf.contrib.layers.batch_norm(output, scale=True, is_training=self.is_training,
                                                          updates_collections=None)
                    output = tf.nn.relu(output)
                    kernel = tf.get_variable(name='kernel_$d_%d' % (block, layer),
                                             shape=[3, 3, int(output.get_shape()[-1]), growth_rate],
                                             initializer=tf.contrib.layers.variance_scaling_initializer())
                    output = tf.nn.conv2d(output, kernel, [1, 1, 1, 1], padding='SAME')
                    output = tf.cond(self.is_training, lambda: tf.nn.dropout(output, self.keep_prob), lambda: output)
                layer_input = tf.concat(values=(layer_input, output), axis=-1)

            # 除了最后一个block，都有transition层
            if block != self.total_blocks - 1:
                out_features = int(int(output.get_shape()[-1]) * self.reduction)
                output = tf.contrib.layers.batch_norm(output, scale=True, is_training=self.is_training,
                                                      updates_collections=None)
                output = tf.nn.relu(output)
                kernel = tf.get_variable(name='kernel_transition_$d' % block,
                                         shape=[1, 1, int(output.get_shape()[-1]), out_features],
                                         initializer=tf.contrib.layers.variance_scaling_initializer())
                output = tf.nn.conv2d(output, kernel, [1, 1, 1, 1], padding='SAME')
                output = tf.cond(self.is_training, lambda: tf.nn.dropout(output, self.keep_prob), lambda: output)
                output = tf.nn.avg_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 最后一个block后面的全局平均池化
        output = tf.contrib.layers.batch_norm(input, scale=True, is_training=self.is_training, updates_collections=None)
        output = tf.nn.relu(output)
        k = int(output.get_shape()[-2])
        output = tf.nn.avg_pool(output, [1, k, k, 1], [1, k, k, 1], padding='VALID')

        # 全连接层 FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W = tf.get_variable(name='W', shape=[features_total, self.n_classes],
                            initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', initializer=tf.constant(0.0, shape=[self.n_classes]))
        logits = tf.nn.bias_add(tf.matmul(output, W), bias)

        prediction = tf.nn.softmax(logits)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
        self.cross_entropy = cross_entropy
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # optimizer and train step
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.train_step = optimizer.minimize(cross_entropy + l2_loss * self.weight_decay)

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        total_start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
            start_time = time.time()
            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                learning_rate = learning_rate / 10
                print("Decrease learning rate, new lr = %f" % learning_rate)

            print("Training...")
            loss, acc = self.train_one_epoch(
                self.data_provider.train, batch_size, learning_rate)
            if self.should_save_logs:
                self.log_loss_accuracy(loss, acc, epoch, prefix='train')

            if train_params.get('validation_set', False):
                print("Validation...")
                loss, acc = self.test(
                    self.data_provider.validation, batch_size)
                if self.should_save_logs:
                    self.log_loss_accuracy(loss, acc, epoch, prefix='valid')

            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if self.should_save_model:
                self.save_model()

        total_training_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(
            seconds=total_training_time)))

    def train_one_epoch(self, data, batch_size, learning_rate):
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            images, labels = batch
            feed_dict = {
                self.images: images,
                self.labels: labels,
                self.learning_rate: learning_rate,
                self.is_training: True,
            }
            fetches = [self.train_step, self.cross_entropy, self.accuracy]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, accuracy = result
            total_loss.append(loss)
            total_accuracy.append(accuracy)
            if self.should_save_logs:
                self.batches_step += 1
                self.log_loss_accuracy(
                    loss, accuracy, self.batches_step, prefix='per_batch',
                    should_print=False)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy

    def test(self, data, batch_size):
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            feed_dict = {
                self.images: batch[0],
                self.labels: batch[1],
                self.is_training: False,
            }
            fetches = [self.cross_entropy, self.accuracy]
            loss, accuracy = self.sess.run(fetches, feed_dict=feed_dict)
            total_loss.append(loss)
            total_accuracy.append(accuracy)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy
