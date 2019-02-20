"""Wasserstein GAN with gradient penalties"""
import os
import sys
import time
import pickle as pkl
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"


def data_batcher(data, batch_size):
    """create yield function for given data and batch size"""

    def get_all_batches():
        """yield function (generator) for all batches in data"""
        # shuffle in place each time
        np.random.shuffle(data)

        # get total number of evenly divisible batchs
        # shape of (num_batches, batch_size, n_features)
        batches = data[:(data.shape[0] // batch_size) * batch_size]
        batches = batches.reshape(-1, batch_size, data.shape[1])

        # go through all batches and yield them
        for i, _ in enumerate(batches):
            yield np.copy(batches[i])

    def infinite_data_batcher():
        """creates a generator that yields new batches every time it is called"""
        # once we run out of batches start over
        while True:
            # for each batch in one set of batches
            for batch in get_all_batches():
                yield batch

    return infinite_data_batcher()


class WGAN():
    """Wasserstein GAN with gradient penalties"""
    params = {
        'base_nodes': 64,
        'critic_iters': 5,  # number of discriminator iterations
        'lambda': 10,  # paramter for gradient penalty
        'num_epochs': 100000  # how long to train for
    }

    def __init__(self, filepath, critic_iters=None, base_nodes=None):
        # set custom options
        if critic_iters:
            self.params['critic_iters'] = critic_iters

        # read in data and split it
        scratch = pd.read_csv(filepath)
        self.col_names = scratch.columns
        data = scratch.values
        train_data, self.test_data = train_test_split(
            data, test_size=0.2, random_state=100)

        self.params['n_features'] = data.shape[1]
        # create 1.5 and 2 times for the generator network dimensions
        self.params['1.5_n_features'] = round(1.5 * self.params['n_features'])
        self.params['2_n_features'] = 2 * self.params['n_features']

        if base_nodes:
            self.params['base_nodes'] = base_nodes
        self.params['2_base_nodes'] = 2 * self.params[
            'base_nodes']  # 2 x base for discriminator
        self.params['4_base_nodes'] = 4 * self.params[
            'base_nodes']  # 4 x base for discriminator

        self.params['n_observations'] = data.shape[0]
        # number of observations divided by the number of critic iterations
        # rounded down to the nearest multiple of 100
        self.params['batch_size'] = int(
            train_data.shape[0] / self.params['critic_iters']) // 100 * 100

        # double check on sizing
        assert self.test_data.shape[0] > self.params['batch_size']
        assert train_data.shape[0] / self.params['batch_size'] > self.params[
            'critic_iters']

        self.train_batcher = data_batcher(train_data,
                                          self.params['batch_size'])

        self.print_settings()

        # predefine values that will be set later
        self.real_data = None
        self.gen_loss = None
        self.disc_loss = None
        self.gen_train_op = None
        self.disc_train_op = None
        self.fixed_noise_samples = None

        # define lists to store data
        self.disc_loss_all = []
        self.gen_loss_all = []
        self.disc_loss_test_all = []
        self.time_all = []

    def print_settings(self):
        """print the settings"""
        for k, v in self.params.items():
            print(f'{k + ":":18}{v}')
        print()

    def generator(self, inpt):
        """create the generator graph"""
        # first dense layer
        output = tf.contrib.layers.fully_connected(
            inpt,
            self.params['2_n_features'],
            activation_fn=tf.nn.relu,
            scope='Generator.1',
            reuse=tf.AUTO_REUSE)

        # second dense layer
        output = tf.contrib.layers.fully_connected(
            output,
            self.params['1.5_n_features'],
            activation_fn=tf.nn.relu,
            scope='Generator.2',
            reuse=tf.AUTO_REUSE)

        # third dense layer
        output = tf.contrib.layers.fully_connected(
            output,
            self.params['n_features'],
            activation_fn=tf.nn.sigmoid,
            scope='Generator.3',
            reuse=tf.AUTO_REUSE)

        return output

    def discriminator(self, output):
        """create the discriminator graph"""
        # create first dense layer
        output = tf.contrib.layers.fully_connected(
            output,
            self.params['base_nodes'],
            activation_fn=tf.nn.leaky_relu,
            scope='Discriminator.1',
            reuse=tf.AUTO_REUSE)

        # create second dense layer
        output = tf.contrib.layers.fully_connected(
            output,
            self.params['2_base_nodes'],
            activation_fn=tf.nn.leaky_relu,
            scope='Discriminator.2',
            reuse=tf.AUTO_REUSE)

        # create third dense layer
        output = tf.contrib.layers.fully_connected(
            output,
            self.params['4_base_nodes'],
            activation_fn=tf.nn.leaky_relu,
            scope='Discriminator.3',
            reuse=tf.AUTO_REUSE)

        # create fourth dense layer
        output = tf.contrib.layers.fully_connected(
            output,
            1,
            activation_fn=None,
            scope='Discriminator.4',
            reuse=tf.AUTO_REUSE)

        return output

    def create_graph(self):
        """create computation graph"""
        # create the placeholder for real data and generator for fake
        self.real_data = tf.placeholder(
            tf.float32,
            shape=[self.params['batch_size'], self.params['n_features']])
        # create a noise data set of size of the number of samples by 100
        noise = tf.random_normal([self.params['batch_size'], 100])
        fake_data = self.generator(noise)

        # run the discriminator for both types of data
        disc_real = self.discriminator(self.real_data)
        disc_fake = self.discriminator(fake_data)

        # create the loss for generator and discriminator
        self.gen_loss = -tf.reduce_mean(disc_fake)
        self.disc_loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        # add the gradient penalty to disc loss
        # create random split of data
        alpha = tf.random_uniform(
            shape=[self.params['batch_size'], 1], minval=0, maxval=0)

        # combine real and fake
        interpolates = (alpha * self.real_data) + ((1 - alpha) * fake_data)
        # compute gradients of dicriminator values
        gradients = tf.gradients(
            self.discriminator(interpolates), [interpolates])[0]
        # calculate the 2 norm of the gradients
        slopes = tf.sqrt(
            tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        # subtract 1, square, use lambda parameter to scale
        gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
        self.disc_loss += self.params['lambda'] * gradient_penalty

        # use adam optimizer on losses
        gen_params = [
            v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if 'Generator' in v.name
        ]
        self.gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
                self.gen_loss, var_list=gen_params)
        disc_params = [
            v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if 'Discriminator' in v.name
        ]
        self.disc_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
                self.disc_loss, var_list=disc_params)

        # for generating samples
        fixed_noise = tf.constant(
            np.random.normal(size=(100000, 100)).astype('float32'))
        self.fixed_noise_samples = self.generator(fixed_noise)

        # with tf.Session() as session:
        #     _ = tf.summary.FileWriter('./logs_new', session.graph)

    def train(self):
        """run the training loop"""
        # saver object for saving the model
        saver = tf.train.Saver()

        with tf.Session() as session:
            # initialize variables
            session.run(tf.global_variables_initializer())

            for epoch in range(self.params['num_epochs']):
                start_time = time.time()

                disc_loss_list = []
                for i in range(self.params['critic_iters']):
                    # get a batch
                    train = next(self.train_batcher)
                    # run one critic iteration
                    disc_loss, _ = session.run(
                        [self.disc_loss, self.disc_train_op],
                        feed_dict={self.real_data: train})
                    print(f'DISC LOSS ITER {i:>3}: {disc_loss}')
                    disc_loss_list.append(disc_loss)

                # run one generator train iteration
                gen_loss, _ = session.run([self.gen_loss, self.gen_train_op])

                # save the loss and time of iteration
                self.time_all.append(time.time() - start_time)
                self.disc_loss_all.append(disc_loss_list)
                self.gen_loss_all.append(gen_loss)

                if epoch < 10 or epoch % 100 == 99:
                    # print the results
                    print((f'Epoch: {epoch:5} '
                           f'[D loss: {self.disc_loss_all[-1][-1]:7.4f}] '
                           f'[G loss: {self.gen_loss_all[-1]:7.4f}] '
                           f'[Time: {self.time_all[-1]:4.2f}]'))

                # if at epoch ending 9999 check test loss
                if epoch == 0 or epoch % 1000 == 999:
                    # shuffle test in place
                    np.random.shuffle(self.test_data)
                    test_disc_loss = session.run(
                        self.disc_loss,
                        feed_dict={
                            self.real_data:
                            self.test_data[:self.params['batch_size']]
                        })
                    self.disc_loss_test_all.append(test_disc_loss)
                    print(
                        f'Test Epoch: [Test D loss: {self.disc_loss_test_all[-1]:7.4f}]'
                    )

                # if at epoch ending 99999 generate large
                if epoch % 10000 == 9999:
                    for i in range(10):
                        samples = session.run(self.fixed_noise_samples)
                        samples = pd.DataFrame(samples, columns=self.col_names)
                        samples.to_csv(
                            f'data/samples_new_{epoch}_synthetic_{i}.csv',
                            index=False)

                # update log every 100
                if epoch < 5 or epoch % 100 == 99:
                    with open(
                            f'log_{self.params["critic_iters"]}_{self.params["base_nodes"]}.pkl',
                            'wb') as f:
                        pkl.dump({
                            'time': self.time_all,
                            'disc_loss': self.disc_loss_all,
                            'gen_loss': self.gen_loss_all,
                            'test_loss': self.disc_loss_test_all
                        }, f)

                if epoch == 500:
                    break

            saver.save(session, os.path.join(os.getcwd(), 'model.ckpt'))


if __name__ == '__main__':
    # parse arguments

    # start with a fresh graph
    tf.reset_default_graph()
    # create object
    wgan = WGAN(
        filepath='data/mimic_train_sdv.csv',
        critic_iters=int(sys.argv[1]),
        base_nodes=int(sys.argv[2]))
    # define the computation graph
    wgan.create_graph()
    # train the model
    wgan.train()
