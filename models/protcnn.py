import tensorflow as tf

from functools import partial
import numpy as np

from tensorflow.keras import backend as K
from keras import objectives, losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda

from utils import aa_letters, luxa_seq

from utils.metrics import aa_acc
from utils.data_loaders import to_one_hot
from utils.decoding import _decode_nonar, batch_temp_sample

nchar = len(aa_letters)


#
# batch_size = 32
# latent_dim = 50


def sampler(latent_dim, epsilon_std=1):
    # [[1.0, 2.0], [3.0, 4.0]]
    # convert_to_tensor thành:
    # [[1. 2.]
    #  [3. 4.]], shape = 2, dtype=float32

    _sampling = lambda z_args: (z_args[0] + K.sqrt(tf.convert_to_tensor(z_args[1] + 1e-8, np.float32)) *
                                K.random_normal(shape=K.shape(z_args[0]), mean=0., stddev=epsilon_std))
    # random_normal: hàm phân phối chuẩn

    return Lambda(_sampling, output_shape=(latent_dim,))


# def sampling(args, epsilon_std=1):
#     z_mean, z_log_var = args
#     epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
#                               stddev=epsilon_std)
#     return z_mean + K.exp(z_log_var / 2) * epsilon

#
# def luxa_batch_conds(n_samples, solubility_level):
#     target_conds = [0, 0, 0]
#     target_conds[solubility_level] = 1
#     target_conds = np.repeat(np.array(target_conds).reshape((1, 3)), n_samples, axis=0)
#     return target_conds


# reshape: định hình lại
# a = np.arange(6).reshape((3, 2))
# => array([[0, 1],
#        [2, 3],
#        [4, 5]])

# np.reshape(a, (2, 3)) # C-like index ordering
# array([[0, 1, 2],
#        [3, 4, 5]])

# repeat: lặp lại
# n_samples số lần lặp lại
# Ví dụ: arr :
#  [[0 1 2]
#  [3 4 5]]
#
# Repeating arr :
#  [[0 0 1 1 2 2]
#  [3 3 4 4 5 5]]

class BaseProtVAE:
    # Child classes must define a self.E, self.G
    def __init__(self,
                 lr=0.001, clipnorm=0., clipvalue=0.,
                 metrics=['accuracy'],
                 # lr=0.001, clipnorm=0., clipvalue=0., metrics=['accuracy', aa_acc],
                 latent_dim=50, seqlen=1653):

        self.latent_dim = latent_dim
        self.seqlen = seqlen

        self.S = sampler(latent_dim, epsilon_std=1.)

        prot = self.E.inputs[0]
        encoder_inp = [prot]
        vae_inp = [prot]

        z_mean, z_var = self.E(encoder_inp)
        # z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_var])
        z = self.S([z_mean, z_var])  # sampler
        self.stochastic_E = Model(inputs=encoder_inp, outputs=[z_mean, z_var, z])

        decoder_inp = [z]
        decoded = self.G(decoder_inp)

        self.VAE = Model(inputs=vae_inp, outputs=decoded)

        def xent_loss(x, x_d_m):  # reconstruction loss
            return K.sum(objectives.categorical_crossentropy(x, x_d_m), -1)

        def kl_loss(x, x_d_m):  # KL divergence loss
            return - 0.5 * K.sum(1 + K.log(z_var + 1e-8) - K.square(z_mean) - z_var, axis=-1)

        def vae_loss(x, x_d_m):
            return xent_loss(x, x_d_m) + kl_loss(x, x_d_m)

        # def vae_loss(x, x_decoded_mean):
        #     xent_loss = original_dim * objectives.categorical_crossentropy(x, x_decoded_mean)
        #     kl_loss = - 0.5 * K.sum(1 + z_var - K.square(z_mean) - K.exp(z_var), axis=-1)
        #     return xent_loss + kl_loss

        log_metrics = metrics + [xent_loss, kl_loss, vae_loss]
        # log_metrics = metrics + [vae_loss]

        print('Learning rate ', lr)
        self.VAE.compile(loss=vae_loss, optimizer=Adam(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue),
                         metrics=log_metrics)
        # self.VAE.compile(loss=vae_loss, optimizer=Adam(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue),
        #                  metrics=log_metrics)
        # self.metric_names = metric_names = ['loss'] + [m.name if type(m) != str else m for m in self.VAE.metrics]
        # self.metric_names = ['loss'] + [m.__name__ if type(m) != str else m for m in self.VAE.metrics]
        print('Protein VAE initialized !')

    def load_weights(self, file='generative_models/weights/default.h5'):
        self.VAE.load_weights(file)
        print('Weights loaded !')
        return self

    def save_weights(self, file='generative_models/weights/default.h5'):
        self.VAE.save_weights(file)
        print('Weights saved !')
        return self

    def prior_sample(self, n_samples=1, mean=0, stddev=1, batch_size=5000):

        # Two - by - four array of samples from N (3, 6.25):
        # 3 + 2.5 * np.random.randn(2, 4)
        # array([[-4.49401501, 4.00950034, -1.81814867, 7.29718677],  # random
        #        [0.39924804, 4.68456316, 4.99394529, 4.84057254]])  # random

        if n_samples > batch_size:
            x = []
            total = 0
            while total < n_samples:
                this_batch = min(batch_size, n_samples - total)
                z_sample = mean + stddev * np.random.randn(this_batch, self.latent_dim)
                # sinh ra ma trận 2 chiều this_batch dòng, latent_dim cột
                x += self.decode(z_sample)
                total += this_batch
        else:
            z_sample = mean + stddev * np.random.randn(n_samples, self.latent_dim)
            x = self.decode(z_sample)
        return x

    def decode(self, z):
        return _decode_nonar(self.G, z)
