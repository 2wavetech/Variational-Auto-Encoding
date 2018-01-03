# Variational-Auto-Encoding
This repositary contains complete Python code implementing VAE. The implementation is based on Keras with TensoFlow as the backend, and uses MNIST dataset for training and result demonstrations.

The core part of my implementation of VAE comes from https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/ but has 3 differences:
1. I have added detailed descriptions of the model and the code.
2. I fixed a bug in the original code of the model at the above URL.
3. More importantly, the URL above does not provide complete Python code that can be run for not only training the model but also demonstrating the intermediate results of encoding and decoding part of VAE. My code here is a complete implementation of VAE.

For those details, please read VAE.md.

My implementation of VAE differes from that of Stanford University (https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py) primarily in the way the intermediate results are produced (one-by-one vs. all at once) and demonstrated, and also in the implementation of VAE loss function (mine is copied from Kristiadi's above, which I think is more straightforward). I found that it does not make a lot of differences between 'adam' that I used and 'rmsprop' that Stanford U. used as optimizer. 

The full-connection network used in the encoder may be replaced by CNN, which I think would perform better and I'd like to try in the future.
