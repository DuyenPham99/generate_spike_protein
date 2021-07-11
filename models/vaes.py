from models.encoders import fc_encoder
from models.protcnn import BaseProtVAE
from models.decoders import fc_decoder


class MSAVAE(BaseProtVAE):
    def __init__(self, latent_dim=50, seqlen=1653, activation='relu',
                 encoder_kwargs={'encoder_hidden': [250, 250],
                                 'encoder_dropout': [0.7, 0.]},
                 decoder_kwargs={'decoder_hidden': [250, 250],
                                 'decoder_dropout': [0., 0.7]}):
        self.E = fc_encoder(seqlen, latent_dim,
                            activation=activation,
                            **encoder_kwargs)
        self.G = fc_decoder(latent_dim, seqlen,
                            activation=activation,
                            **decoder_kwargs)
        super().__init__(latent_dim=latent_dim, seqlen=seqlen)
