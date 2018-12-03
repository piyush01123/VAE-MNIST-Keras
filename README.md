
## VAE in Keras
#### Model Architecture:
Input(784) --> Dense(512) --> 2*Dense(2) --> z_mean(2), z_var(2) --> Latent(2) by Reparameterization Trick
Latent(2) -->  Dense(512) --> Dense(784)

<img src="vae_mlp_encoder.png"/>

<img src="vae_mlp_decoder.png"/>
## Tensorboard
Tensorboard can be run using `tensorflow.python.keras.callbacks.TensorBoard` API.


### Reparameterization
Reparameterization Trick is a crucial part of VAE as mentioned in Kingma's paper [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf).

Nice visual of how reparameterization helps in calculating gradient of a stochastic objective function:
<http://nbviewer.jupyter.org/github/gokererdogan/Notebooks/blob/master/Reparameterization%20Trick.ipynb>.
