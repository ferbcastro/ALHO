# Autoencoder

The autoencoder is made up of two networks: an encoder and decoder. The idea is that the encoder will
receive the features as input as encode them into a vector of some latent dimension of our choice, the decoder
will be useful through the training process to check if the representation is useful.

## What type of autoencoder?

Theres a bunch of autoencoders uncomplete, sparse, convolutional... While not idea for the task we will start
with a undercomplete autoencoder for simplicity, what that means is that the model lacks regularization 
mechanisms that prevent some forms overfitting, like a sparsity penalty to make the model not act as map.

This will be improved later.

For the especific implementation of the encoder we will use classical feed forward networks, the same ones
on the 3b1b video.  What does that mean? it means that is just a bunch of linear layers and a activation 
fucntion stacked on top of each other. A linear layer just means it uses the standard matrix multiplication
plus bias, then the non linearity is added later when the activation function like sigmoid or ReLU is applied.

## Hyperparemeters

I thinks this part seems more of an art than science lets see

### Activation Function 

[Source](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)

#### Hidden Layers

Well theres a three options but not really. Theres ReLU,
sigmoid and tanh, while they serve the same purpose
both sigmoid and tanh experience a thing called 
vanishing gradient (dont ask) so today the main 
recomendation is to always use ReLU for the hidden layers.
Although it seems tanh is still used on recurrent nets.

Theres a problem on ReLU called dead units or something
but if we see that happening theres a variation that fixes
that. We switch if we start seeing too many neurons stuck
at zero. We wont use LeakyRelu from the get go because 
it introduces new things to optimize.

#### Output Layer

The encoder will have a linear output layer, apparently
it makes it learn a more meaningful represesntation
since they can take any real value. 

The decoder will be a sigmoid will be used as the 
the vanishing gradient doesnt seem to be a big problem
on the output layer(TODO investigate why), also on [An Introduction to Autoencoders](https://arxiv.org/pdf/2201.03898)
is recommended to use sigmoid when the date is between
0 and 1, our case.

### Layers

[The correct way](https://docs.pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)
the proper way to do this choice is by doing a hyperparameter
search but right now we are aiming for a simpler aproach
 
It seems that most autoencoder are simetric meaning the 
decoder is just the encoder mirrored so lets follow this 
paradigm. Still we have to decide on a initial setup for 
the network. 

I thinks the best aproach for the start is just to go down
by half on each layer until the desired latent dim, this 
way we avoid abrupt compression and lose data.

### Latent Space (aka oq importa) 

Choose a 1000, because my heart told to

## The code

Surprisingly simple

## nn.Module

This baby is our 2x2 lego piece, what it contains is the 
actual weights and biases. The idea of the code is that 
as you define your net everything is always inherited from
nn.Module, so for example if you define a linear layers
that is a nn.Module and the weights and bias of that layers
will be tracked by ther parent objects as they are 
recursively registered when the class is instantiated( 
this is done by the call to the parent contructor), 
so they can be passed to the optimizer later on.

## forward()

This method should always be overwritten, it basically
tells the computation that the model performs, so as
an example on the autoencoder we will call the encoder
computation then the decoder computation and return the 
result

