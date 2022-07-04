# Variational autoencoder for MNIST

## Train, Test
```
python src/mnist-vae/train.py
python src/mnist-vae/test.py
```

`test.py` samples from the trained VAE. The results are shown in `src/mnist-vae/img/sample.png`.

## Footnote
This VAE is based off [this post](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/). $p(x\mid z)$ is modelled as a Bernoulli distribution.
