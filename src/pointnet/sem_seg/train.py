import argparse
import sys
import os
import re
import numpy as np

import tensorflow as tf

from pointnet import provider
from pointnet.sem_seg.model import get_model, get_loss, placeholder_inputs

class Trainer:
    def __init__(self, gpu=0, log_dir='log', num_point=4096, max_epoch=50, batch_size=24,
                 learning_rate=0.001, momentum=0.9, optimizer='adam',
                 decay_step=300000, decay_rate=0.5, test_area=6,
                 data_path='indoor3d_sem_seg_hdf5_data'):

        self._batch_size = batch_size
        self._num_point = num_point
        self._max_epoch = max_epoch
        self._base_learning_rate = learning_rate
        self._gpu_index = gpu
        self._momentum = momentum
        self._optimizer = optimizer
        self._decay_step = decay_step
        self._decay_rate = decay_rate

        self._log_dir = log_dir
        if not os.path.exists(self._log_dir):
            os.mkdir(self._log_dir)

        self._log_fout = open(os.path.join(self._log_dir, 'log_train.txt'), 'w')

        #self._max_num_point = 4096
        self._num_classes = 13
        self._bn_init_decay = 0.5
        self._bn_decay_decay_rate = 0.5
        #self._bn_decay_decay_step = float(_decay_step * 2)
        self._bn_decay_decay_step = float(self._decay_step)
        self._bn_decay_clip = 0.99

        all_files = sorted([os.path.join(data_path, i) for i in os.listdir(data_path)
                            if re.match(r'ply_data_all_(\d+).h5', i)])

        room_filelist = [line.rstrip() for line in open(os.path.join(data_path, 'room_filelist.txt'))]

        # Load ALL data
        data_batch_list = []
        label_batch_list = []
        for h5_filename in all_files:
            data_batch, label_batch = provider.load_h5(h5_filename)
            data_batch_list.append(data_batch)
            label_batch_list.append(label_batch)
        data_batches = np.concatenate(data_batch_list, 0)
        label_batches = np.concatenate(label_batch_list, 0)

        test_area = f'Area_{test_area}'
        train_idxs = []
        test_idxs = []
        for i,room_name in enumerate(room_filelist):
            if test_area in room_name:
                test_idxs.append(i)
            else:
                train_idxs.append(i)

        self.train_data = data_batches[train_idxs,...]
        self.train_label = label_batches[train_idxs]
        self.test_data = data_batches[test_idxs,...]
        self.test_label = label_batches[test_idxs]


    def log_string(self, out_str):
        self._log_fout.write(out_str + '\n')
        self._log_fout.flush()
        print(out_str)


    def get_learning_rate(self, batch):
        learning_rate = tf.train.exponential_decay(
                            self._base_learning_rate,  # Base learning rate.
                            batch * self._batch_size,  # Current index into the dataset.
                            self._decay_step,          # Decay step.
                            self._decay_rate,          # Decay rate.
                            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
        return learning_rate

    def get_bn_decay(self, batch):
        bn_momentum = tf.train.exponential_decay(
                          self._bn_init_decay,
                          batch*self._batch_size,
                          self._bn_decay_decay_step,
                          self._bn_decay_decay_rate,
                          staircase=True)
        bn_decay = tf.minimum(self._bn_decay_clip, 1 - bn_momentum)
        return bn_decay

    def train(self):
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(self._gpu_index)):
                pointclouds_pl, labels_pl = placeholder_inputs(self._batch_size, self._num_point)
                is_training_pl = tf.placeholder(tf.bool, shape=())

                # Note the global_step=batch parameter to minimize.
                # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
                batch = tf.Variable(0)
                bn_decay = self.get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)

                # Get model and loss
                pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
                loss = get_loss(pred, labels_pl)
                tf.summary.scalar('loss', loss)

                correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
                accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(self._batch_size * self._num_point)
                tf.summary.scalar('accuracy', accuracy)

                # Get training operator
                learning_rate = self.get_learning_rate(batch)
                tf.summary.scalar('learning_rate', learning_rate)
                if self._optimizer == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self._momentum)
                elif self._optimizer == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                else:
                    raise NotImplementedError(f'Optimizer {self._optimizer} not supported')

                train_op = optimizer.minimize(loss, global_step=batch)

                # Add ops to save and restore all the variables.
                saver = tf.train.Saver()

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = True
            sess = tf.Session(config=config)

            # Add summary writers
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(self._log_dir, 'train'),
                                                 sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(self._log_dir, 'test'))

            # Init variables
            init = tf.global_variables_initializer()
            sess.run(init, {is_training_pl:True})

            ops = {'pointclouds_pl': pointclouds_pl,
                   'labels_pl': labels_pl,
                   'is_training_pl': is_training_pl,
                   'pred': pred,
                   'loss': loss,
                   'train_op': train_op,
                   'merged': merged,
                   'step': batch}

            for epoch in range(self._max_epoch):
                self.log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()

                self.train_one_epoch(sess, ops, train_writer)
                self.eval_one_epoch(sess, ops, test_writer)

                # Save the variables to disk.
                if epoch % 10 == 0:
                    save_path = saver.save(sess, os.path.join(self._log_dir, "model.ckpt"))
                    self.log_string("Model saved in file: %s" % save_path)



    def train_one_epoch(self, sess, ops, train_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = True

        self.log_string('----')
        current_data, current_label, _ = provider.shuffle_data(self.train_data[:, 0:self._num_point, :],
                                                               self.train_label)

        file_size = current_data.shape[0]
        num_batches = file_size // self._batch_size

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(num_batches):
            if batch_idx % 100 == 0:
                print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx+1) * self._batch_size

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                             feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += (self._batch_size * self._num_point)
            loss_sum += loss_val

        self.log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        self.log_string('accuracy: %f' % (total_correct / float(total_seen)))


    def eval_one_epoch(self, sess, ops, test_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = False
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(self._num_classes)]
        total_correct_class = [0 for _ in range(self._num_classes)]

        self.log_string('----')
        current_data = self.test_data[:, 0:self._num_point, :]
        current_label = np.squeeze(self.test_label)

        file_size = current_data.shape[0]
        num_batches = file_size // self._batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx+1) * self._batch_size

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                          feed_dict=feed_dict)
            test_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += (self._batch_size * self._num_point)
            loss_sum += (loss_val * self._batch_size)
            for i in range(start_idx, end_idx):
                for j in range(self._num_point):
                    l = current_label[i, j]
                    total_seen_class[l] += 1
                    total_correct_class[l] += (pred_val[i-start_idx, j] == l)

        self.log_string('eval mean loss: %f' % (loss_sum / float(total_seen / self._num_point)))
        self.log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
        self.log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))

    def close(self):
        self._log_fout.close()

def start_debug():
    import pydevd
    pydevd.settrace('localhost', port=12151, stdoutToServer=True, stderrToServer=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
    parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
    parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
    parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 6]')
    parser.add_argument('--data_path', type=str, default='indoor3d_sem_seg_hdf5_data', help='Data path where .h5 files are found')
    FLAGS = parser.parse_args()

    trainer = Trainer(**FLAGS.__dict__)

    trainer.train()
    trainer.close()

if __name__ == "__main__":
    main()
