# External imports
import torch 
import torch.nn as nn
import torch.nn.functional as F

# Internal imports
from alpha import Alpha
from hierarchical_operation import HierarchicalOperation

class BetaVAE(nn.Module):
    '''
    Beta VAE - https://openreview.net/forum?id=Sy2fzU9gl
    '''
    def __init__(self, alpha: Alpha, beta: float, primitives: dict, channels_in: int, image_height: int, image_width: int, latent_size=32, writer=None, test_mode=False):
        '''
        Input: 
        - alpha - an object of type Alpha
        - beta - from Beta VAE paper, weight to put on KL divergence part of ELBO loss - biases towards disentangled representations
        - primitives - dict[any -> lambda function with inputs C, stride, affine that returns a primitive operation]
        - channels_in - the input channels from the dataset
        - latent_size - 
        '''

        # Superclass constructor
        super().__init__()

        # Initialize member variables
        self.alpha = alpha
        self.beta = beta
        self.image_height = image_height
        self.image_width = image_width
        self.writer = writer
        self.test_mode = test_mode

        '''
        Encoder: Top-Level DAG of HDARTS used here
        '''
        # Dict from edge tuple to MixedOperation on that edge
        self.encoder = HierarchicalOperation.create_dag(
            level=alpha.num_levels - 1,
            alpha=alpha,
            alpha_dag=alpha.parameters[alpha.num_levels - 1][0],
            primitives=primitives,
            channels_in=channels_in     
        )

        self.flattened_feature_count = self.encoder.channels_out * image_height * image_width
        self.fc_mu = nn.Linear(self.flattened_feature_count, latent_size)
        self.fc_var = nn.Linear(self.flattened_feature_count, latent_size)
        
        '''
        Decoder: Static Currently (TODO: Implement reverse of encoder)
        '''
        self.fc_z = nn.Linear(latent_size, self.flattened_feature_count)
        self.decoder =  nn.Sequential(
            self._deconv(self.encoder.channels_out, 64),
            self._deconv(64, 32),
            self._deconv(32, 32, 1),
            self._deconv(32, 3),
            nn.Sigmoid()
        )

    def _deconv(self, in_channels, out_channels, out_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, output_padding=out_padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.flattened_feature_count)
        return self.fc_mu(x), self.fc_var(x)
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(-1, self.encoder.channels_out, self.image_height, self.image_width)
        return self.decoder(z)

    def forward(self, x): 
        if not self.test_mode:
            if (torch.cuda.is_available()):
                x = x.cuda()
               
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return (recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size