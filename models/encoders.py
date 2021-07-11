from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout, Flatten
from utils import aa_letters


def fc_encoder(seqlen, latent_dim, alphabet_size=len(aa_letters), encoder_hidden=[250, 250],
               encoder_dropout=[0.7, 0.], activation='relu'):
    x = Input(shape=(seqlen, alphabet_size,))  #
    h = Flatten()(x)

    # flatten: làm phẳng
    # A =  [[[47 83]
    #   [38 53]
    #   [76 24]
    #   [15 49]]
    #
    #  [[23 26]
    #   [30 43]
    #   [30 26]
    #   [58 92]]
    #
    #  [[69 80]
    #   [73 47]
    #   [50 76]
    #   [37 34]]]
    #
    # Flattened_X =  [47 83 38 53 76 24 15 49 23 26 30 43 30 26 58 92 69 80 73 47 50 76 37 34]

    # x =  [110 202]
    # y =  [108  70   6]
    # c = np.concatenate((x,y))
    # c =  [110 202 108  70   6  10]

    for n_hid, drop in zip(encoder_hidden, encoder_dropout):
        # numberList = [1, 2, 3]
        # strList = ['one', 'two', 'three']
        # result = zip(numberList, strList)
        # => result {(2, 'two'), (3, 'three'), (1, 'one')}
        h = Dense(n_hid, activation=activation)(h)
        if drop > 0:
            h = Dropout(drop)(h)

    # Variational parameters
    z_mean = Dense(latent_dim)(h)
    z_var = Dense(latent_dim, activation='softplus')(h)
    E = Model(x, [z_mean, z_var])
    return E
