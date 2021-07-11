from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Dense, Activation, Dropout
from utils import aa_letters


def fc_decoder(latent_dim, seqlen, decoder_hidden=[250, 250], decoder_dropout=[0., 0.7],
               alphabet_size=len(aa_letters), activation='relu'):
    latent_vector = Input((latent_dim,))
    decoder_x = latent_vector

    for h, d in zip(decoder_hidden, decoder_dropout):
        decoder_x = Dense(h, activation=activation)(decoder_x)
        if d > 0:
            decoder_x = Dropout(d)(decoder_x)

    decoder_out = Dense(seqlen * alphabet_size, activation=None)(decoder_x)
    decoder_out = Reshape((seqlen, alphabet_size))(decoder_out)
    decoder_out = Activation('softmax')(decoder_out)

    G = Model(latent_vector, decoder_out)
    return G
