import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule

from pyro.nn import PyroSample

### Class to make a Bayesian Binary Classifier
class BayesianBinaryClassifier(PyroModule):
    def __init__(self, layers, activation=F.relu):
        super(BayesianBinaryClassifier, self).__init__()

        self.layers = layers
        self.activation = activation

        ### I define probabilistic layers for each of them
        for i in range(len(layers)-1):
            setattr(self,"layer{}".format(i), PyroModule[nn.Linear](layers[i], layers[i+1]))
            self.__getattr__("layer{}".format(i)).weight= PyroSample(dist.Normal(0., 1.).expand([layers[i+1], layers[i]]).to_event(2))
            self.__getattr__("layer{}".format(i)).bias = PyroSample(dist.Normal(0., 1.).expand([layers[i+1]]).to_event(1))



        #### The last layer for the binary classifier is always
        self.output_layer = PyroModule[nn.Linear](layers[-1], 1)
        self.output_layer.weight = PyroSample(dist.Normal(0., 1).expand([1, layers[-1]]).to_event(2))
        self.output_layer.bias = PyroSample(dist.Normal(0., 1).expand([1]).to_event(1))

    def forward(self, x, y=None):
        out = x
        for i in range(len(self.layers)-1):
            out = self.__getattr__("layer{}".format(i))(out)
            out = self.activation(out)


        #### The layers before give the mean
        logit = self.output_layer(out).squeeze(-1)

        ## in case I provide a y I want to condition on data
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Bernoulli(logits=logit), obs=y)

        return torch.sigmoid(logit)


### Class to make bayesian classifier
class BayesianClassifier(PyroModule):
    def __init__(self, layers, num_categories, activation=F.relu):
        super(BayesianClassifier, self).__init__()

        self.layers = layers
        self.num_categories = num_categories
        self.activation = activation

        ### I define probabilistic layers for each of them
        for i in range(len(layers)-1):
            setattr(self,"layer{}".format(i), PyroModule[nn.Linear](layers[i], layers[i+1]))
            self.__getattr__("layer{}".format(i)).weight= PyroSample(dist.Normal(0., 1.).expand([layers[i+1], layers[i]]).to_event(2))
            self.__getattr__("layer{}".format(i)).bias = PyroSample(dist.Normal(0., 1.).expand([layers[i+1]]).to_event(1))

        ### The output layer should be separate since it has a different activation function
        self.output_layer = PyroModule[nn.Linear](layers[-1], num_categories)
        self.output_layer.weight = PyroSample(dist.Normal(0., 1.).expand([num_categories, layers[-1]]).to_event(2))
        self.output_layer.bias = PyroSample(dist.Normal(0., 1.).expand([num_categories]).to_event(1))

    def forward(self, x, y=None):
        out = x
        for i in range(len(self.layers)-1):
            out = self.__getattr__("layer{}".format(i))(out)
            out = self.activation(out)

        mean_logit = self.output_layer(out)

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=mean_logit), obs=y)

        return torch.softmax(mean_logit, axis=1)



""" 
    def gpu(self):
        for i in range(len(self.layers)-1):
            self.__setattr__("layer{}".format(i), self.__getattr__("layer{}".format(i)).cuda())
            self.__getattr__("layer{}".format(i)).weight= self.__getattr__("layer{}".format(i)).weight.cuda()
            self.__getattr__("layer{}".format(i)).bias = self.__getattr__("layer{}".format(i)).bias.cuda()

        self.output_layer = self.output_layer.cuda()
        self.output_layer.weight = self.output_layer.weight.cuda()
        self.output_layer.bias = self.output_layer.bias.cuda()

    def cpu(self):
        for i in range(len(self.layers) - 1):
            self.__setattr__("layer{}".format(i), self.__getattr__("layer{}".format(i)).cpu())
            self.__getattr__("layer{}".format(i)).weight = self.__getattr__("layer{}".format(i)).weight.cpu()
            self.__getattr__("layer{}".format(i)).bias = self.__getattr__("layer{}".format(i)).bias.cpu()
 
        self.output_layer = self.output_layer.cpu()
        self.output_layer.weight = self.output_layer.weight.cpu()
        self.output_layer.bias = self.output_layer.bias.cpu()
"""












