# Simple and Effective VAE training with σ-VAE in PyTorch

[[Project page]](https://orybkin.github.io/sigma-vae/) [[Colab]](https://colab.research.google.com/drive/1mQr1SkiSQLhCSsVaj4R7XLcinknwHiV8?usp=sharing) [[TensorFlow implementation]](https://github.com/orybkin/sigma-vae-tensorflow) 

This is the PyTorch implementation of the σ-VAE paper. See the σ-VAE project page for more info, results, and alternative
 implementations. Also see the Colab version of this repo to train a sigma-VAE with zero setup needed!

This implementation is based on the VAE from PyTorch [examples](https://github.com/pytorch/examples/blob/master/vae/main.py). In contrast to the original implementation,  the σ-VAE 
achieves good results without tuning the heuristic weight beta since the decoder variance balances the objective. 
It is also very easy to implement, check out individual commits to see the few lines of code you need to add this to your VAE.!

## How to run it 

This repo implements several VAE versions.

First, a VAE from the original PyTorch example repo that uses MSE loss. This implementation works very poorly because
the MSE loss averages the pixels instead of summing them. Don't do this! You have to sum the loss across pixels and
latent dimensions according to the definition of multivariate Gaussian (and other) distributions.
```
python train_vae.py --log_dir mse_vae --model mse_vae
```

Summing the loss works a bit better and is equivalent to the Gaussian negative log likelihood (NLL) with a certain, constant 
variance. This second model uses the Gaussian NLL as the reconstruction term. However, since the variance is constant
it is still unable to balance the reconstruction and KL divergence term.
```
python train_vae.py --log_dir gaussian_vae --model gaussian_vae
```

The third model is the σ-VAE. It learns the variance of the decoding distribution, which works significantly better and produces
high-quality samples. This is because learning the variance automatically balances the VAE objective. One could balance 
the objective manually by using beta-VAE, however, this is not required when learning the variance!
```
python train_vae.py --log_dir sigma_vae --model sigma_vae
```

Finally, optimal sigma-VAE uses a batch-wise analytic estimate of the variance, which speeds up learning and improves results.
It is also extremely easy to implement! 
```
python train_vae.py --log_dir optimal_sigma_vae --model optimal_sigma_vae
```

