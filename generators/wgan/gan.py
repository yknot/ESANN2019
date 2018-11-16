import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys

import time
import matplotlib
import numpy as np
import tensorflow as tf
import pandas as pd

import tflib as lib
import tflib.ops.linear
import tflib.mimic
import tflib.plot


# settings
sys.path.append(os.getcwd())
matplotlib.use('Agg')

MODE = 'wgan-gp'  # dcgan, wgan, or wgan-gp
DIM = 64  # Model dimensionality
CRITIC_ITERS = 5  # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10  # Gradient penalty lambda hyperparameter
ITERS = 100000  # How many generator iterations to train for


# parse arguments
if len(sys.argv) == 1:
    print('Usage: python gan_mnist.py <data_file> (batch_size)')
    sys.exit(0)

# load data
filepath = sys.argv[1]

if not os.path.isfile(filepath):
    print("Couldn't find MIMIC data")
    raise FileNotFoundError

df = pd.read_csv(filepath).values
OUTPUT_DIM = df.shape[1]  # Number of columns in the data
SIZE = df.shape[0]  # number of observations

if len(sys.argv) == 3:
    BATCH_SIZE = int(sys.argv[2])
else:
    BATCH_SIZE = int(SIZE / CRITIC_ITERS // 100 * 100)

# output model settings
lib.print_model_settings(locals().copy())


def LeakyReLU(x, alpha=0.2):
    """leaky relu function"""
    return tf.maximum(alpha * x, x)


def ReLULayer(name, n_in, n_out, inputs):
    """relu layer"""
    output = lib.ops.linear.Linear(
        name + '.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    """leaky relu layer using function we created"""
    output = lib.ops.linear.Linear(
        name + '.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)


def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 100])

    # first dense layer
    output = lib.ops.linear.Linear(
        'Generator.Input', 100, 2 * OUTPUT_DIM, noise)
    output = tf.nn.relu(output)

    # second dense layer
    output = lib.ops.linear.Linear(
        'Generator.2', 2 * OUTPUT_DIM, round(1.5 * OUTPUT_DIM), output)
    output = tf.nn.relu(output)

    # third dense layer
    output = lib.ops.linear.Linear('Generator.3', round(1.5 * OUTPUT_DIM),
                                   OUTPUT_DIM, output)
    output = tf.nn.sigmoid(output)

    return output


def Discriminator(output):
    output = lib.ops.linear.Linear(
        'Discriminator.1', OUTPUT_DIM, DIM, output)
    output = LeakyReLU(output)

    output = lib.ops.linear.Linear(
        'Discriminator.2', DIM, 2 * DIM, output)
    output = LeakyReLU(output)

    output = lib.ops.linear.Linear(
        'Discriminator.3', 2 * DIM, 4 * DIM, output)
    output = LeakyReLU(output)

    output = lib.ops.linear.Linear(
        'Discriminator.Output', 4 * DIM, 1, output)

    return output


real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var,
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE, 1],
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha * differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(
        tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
    disc_cost += LAMBDA * gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake,
        tf.ones_like(disc_fake)
    ))

    disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake,
        tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_real,
        tf.ones_like(disc_real)
    ))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

# For saving samples
fixed_small_noise = tf.constant(
    np.random.normal(size=(100, 100)).astype('float32'))
fixed_small_noise_samples = Generator(100, noise=fixed_small_noise)

fixed_large_noise = tf.constant(
    np.random.normal(size=(100000, 100)).astype('float32'))
fixed_large_noise_samples = Generator(100000, noise=fixed_large_noise)


def generate_rows(frame, true_dist, small=True):
    if small:
        samples = session.run(fixed_small_noise_samples)
    else:
        samples = session.run(fixed_large_noise_samples)
    np.save('data/samples_{}'.format(frame),
            samples)


train_gen, dev_gen, test_gen = lib.mimic.load(df, BATCH_SIZE, BATCH_SIZE)


def inf_train_gen():
    while True:
        for rows in train_gen():
            yield rows


# config = tf.ConfigProto(inter_op_parallelism_threads=24,
#                         intra_op_parallelism_threads=24,
#                         allow_soft_placement=True, device_count={'CPU': 24})

saver = tf.train.Saver()

# Train loop
with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    gen = inf_train_gen()

    for iteration in range(ITERS):
        start_time = time.time()

        if iteration > 0:
            _ = session.run(gen_train_op)

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data = next(gen)
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 10000 iters
        if iteration % 1000 == 9999:
            dev_disc_costs = []
            for rows in dev_gen():
                _dev_disc_cost = session.run(
                    disc_cost,
                    feed_dict={real_data: rows}
                )
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_rows(iteration, _data)

            if iteration % 10000 == 9999:
                generate_rows(str(iteration) + '_large', _data, small=False)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()

    saver.save(session, os.path.join(os.getcwd(), 'model.ckpt'))

    generate_rows('extra_large', _data, small=False)
