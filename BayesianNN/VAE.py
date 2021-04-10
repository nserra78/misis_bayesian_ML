import torch
import pyro
import pyro.distributions as dist
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, layers, z_dim, activation=nn.Softplus()):
        super(Encoder, self).__init__()

        self.layers = layers
        self.activation = activation
        self.z_dim = z_dim

        for i in range(len(layers)-1):
            self.__setattr__("layer{}".format(i), nn.Linear(layers[i], layers[i+1]))

        self.mean = nn.Linear(layers[-1], z_dim)
        self.log_var = nn.Linear(layers[-1], z_dim)


    def forward(self, x):
        out = x
        for i in range(len(self.layers)-1):
            out = self.__getattr__("layer{}".format(i))(out)
            out = self.activation(out)

        z_loc = self.mean(out)
        z_scale = torch.exp(self.log_var(out))

        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, layers, output_dim, activation=nn.Softplus(), out_activation=nn.Sigmoid()):
        super(Decoder, self).__init__()
        self.activation = activation
        self.z_dim = layers[0]
        self.layers = layers
        self.out_activation = out_activation
        self.output_dim = output_dim

        for i in range(len(layers)-1):
            self.__setattr__("layer{}".format(i), nn.Linear(layers[i], layers[i+1]))

        self.output =  nn.Linear(layers[-1], output_dim)

    def forward(self, x):
        out = x
        for i in range(len(self.layers)-1):
            out = self.__getattr__("layer{}".format(i))(out)
            out = self.activation(out)

        prob = self.out_activation(self.output(out))
        return prob




### This is the vae class that uses encoder, decoder and pyro inference
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def model(self, x, y=None):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.decoder.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.decoder.z_dim)))

            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            probs = self.decoder(z)

            pyro.sample("obs", dist.Bernoulli(probs).to_event(1), obs=y)

            return probs




    def guide(self, x, y=None):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder(x)

            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))


    def reconstruction(self, x):
        z_loc, z_scale = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        probs = self.decoder(z)
        return z, probs






