import argparse
import socket

import sys
from pointnet import provider
from .model import *


class Trainer:
    def __init__(self, FLAGS):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = os.path.dirname(BASE_DIR)

        self.BATCH_SIZE = FLAGS.batch_size
        self.NUM_POINT = FLAGS.num_point
        self.MAX_EPOCH = FLAGS.max_epoch
        self.NUM_POINT = FLAGS.num_point
        self.BASE_LEARNING_RATE = FLAGS.learning_rate
        self.GPU_INDEX = FLAGS.gpu
        self.MOMENTUM = FLAGS.momentum
        self.OPTIMIZER = FLAGS.optimizer
        self.DECAY_STEP = FLAGS.decay_step
        self.DECAY_RATE = FLAGS.decay_rate

        self.LOG_DIR = FLAGS.log_dir
        if not os.path.exists(self.LOG_DIR):
            os.mkdir(self.LOG_DIR)
        os.system('cp model.py %s' % (self.LOG_DIR)) # bkp of model def
        os.system('cp train.py %s' % (self.LOG_DIR)) # bkp of train procedure
        self.LOG_FOUT = open(os.path.join(self.LOG_DIR, 'log_train.txt'), 'w')
        self.LOG_FOUT.write(str(FLAGS)+'\n')

        self.MAX_NUM_POINT = 4096
        self.NUM_CLASSES = 13
        self.BN_INIT_DECAY = 0.5
        self.BN_DECAY_DECAY_RATE = 0.5
        #self.BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
        self.BN_DECAY_DECAY_STEP = float(self.DECAY_STEP)
        self.BN_DECAY_CLIP = 0.99

        self.HOSTNAME = socket.gethostname()

        ALL_FILES = provider.getDataFiles('indoor3d_sem_seg_hdf5_data/all_files.txt')
        room_filelist = [line.rstrip() for line in open('indoor3d_sem_seg_hdf5_data/room_filelist.txt')]

        # Load ALL data
        data_batch_list = []
        label_batch_list = []
        for h5_filename in ALL_FILES:
            data_batch, label_batch = provider.loadDataFile(h5_filename)
            data_batch_list.append(data_batch)
            label_batch_list.append(label_batch)
        data_batches = np.concatenate(data_batch_list, 0)
        label_batches = np.concatenate(label_batch_list, 0)
        print(data_batches.shape)
        print(label_batches.shape)

        test_area = 'Area_'+str(FLAGS.test_area)
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
        print(self.train_data.shape, self.train_label.shape)
        print(self.test_data.shape, self.test_label.shape)



    def log_string(self, out_str):
        self.LOG_FOUT.write(out_str+'\n')
        self.LOG_FOUT.flush()
        print(out_str)


    def get_learning_rate(self, batch):
        learning_rate = tf.train.exponential_decay(
                            self.BASE_LEARNING_RATE,  # Base learning rate.
                            batch * self.BATCH_SIZE,  # Current index into the dataset.
                            self.DECAY_STEP,          # Decay step.
                            self.DECAY_RATE,          # Decay rate.
                            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
        return learning_rate

    def get_bn_decay(self, batch):
        bn_momentum = tf.train.exponential_decay(
                          self.BN_INIT_DECAY,
                          batch*self.BATCH_SIZE,
                          self.BN_DECAY_DECAY_STEP,
                          self.BN_DECAY_DECAY_RATE,
                          staircase=True)
        bn_decay = tf.minimum(self.BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay

    def train(self):
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(self.GPU_INDEX)):
                pointclouds_pl, labels_pl = placeholder_inputs(self.BATCH_SIZE, self.NUM_POINT)
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
                accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(self.BATCH_SIZE*self.NUM_POINT)
                tf.summary.scalar('accuracy', accuracy)

                # Get training operator
                learning_rate = self.get_learning_rate(batch)
                tf.summary.scalar('learning_rate', learning_rate)
                if self.OPTIMIZER == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self.MOMENTUM)
                elif self.OPTIMIZER == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
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
            train_writer = tf.summary.FileWriter(os.path.join(self.LOG_DIR, 'train'),
                                      sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(self.LOG_DIR, 'test'))

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

            for epoch in range(self.MAX_EPOCH):
                self.log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()

                self.train_one_epoch(sess, ops, train_writer)
                self.eval_one_epoch(sess, ops, test_writer)

                # Save the variables to disk.
                if epoch % 10 == 0:
                    save_path = saver.save(sess, os.path.join(self.LOG_DIR, "model.ckpt"))
                    self.log_string("Model saved in file: %s" % save_path)



    def train_one_epoch(self, sess, ops, train_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = True

        self.log_string('----')
        current_data, current_label, _ = provider.shuffle_data(self.train_data[:, 0:self.NUM_POINT, :],
                                                               self.train_label)

        file_size = current_data.shape[0]
        num_batches = file_size // self.BATCH_SIZE

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(num_batches):
            if batch_idx % 100 == 0:
                print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
            start_idx = batch_idx * self.BATCH_SIZE
            end_idx = (batch_idx+1) * self.BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                             feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += (self.BATCH_SIZE * self.NUM_POINT)
            loss_sum += loss_val

        self.log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        self.log_string('accuracy: %f' % (total_correct / float(total_seen)))


    def eval_one_epoch(self, sess, ops, test_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = False
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(self.NUM_CLASSES)]
        total_correct_class = [0 for _ in range(self.NUM_CLASSES)]

        self.log_string('----')
        current_data = self.test_data[:, 0:self.NUM_POINT, :]
        current_label = np.squeeze(self.test_label)

        file_size = current_data.shape[0]
        num_batches = file_size // self.BATCH_SIZE

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.BATCH_SIZE
            end_idx = (batch_idx+1) * self.BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                          feed_dict=feed_dict)
            test_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += (self.BATCH_SIZE * self.NUM_POINT)
            loss_sum += (loss_val * self.BATCH_SIZE)
            for i in range(start_idx, end_idx):
                for j in range(self.NUM_POINT):
                    l = current_label[i, j]
                    total_seen_class[l] += 1
                    total_correct_class[l] += (pred_val[i-start_idx, j] == l)

        self.log_string('eval mean loss: %f' % (loss_sum / float(total_seen / self.NUM_POINT)))
        self.log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
        self.log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    def close(self):
        self.LOG_FOUT.close()


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
    FLAGS = parser.parse_args()

    trainer = Trainer(FLAGS)

    trainer.train()
    trainer.close()

if __name__ == "__main__":
    main()
