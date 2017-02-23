Introductory lecture on deep learning in CS 268 at UCLA.

Install Torch from [http://torch.ch/docs/getting-started.html](http://torch.ch/docs/getting-started.html).

In addition to this, you will need two packages:
- Install the "mnist" package using `luarocks install mnist`
- Install the fakecuda package from [https://github.com/soumith/fakecuda](https://github.com/soumith/fakecuda)

Run the code in this directory as
```
th mnist.lua -h
```

This is a LeNet-style network for MNIST, it should train to about ~1% testing error in 10 epochs and should get to ~0.75% error around 50 epochs.


### Instructions for PyTorch


The file ``mnist.py`` contains code to train a CNN on MNIST using [PyTorch](http://pytorch.org).

1. Installing Python on Mac is easiest with conda: [https://www.continuum.io/downloads](https://www.continuum.io/downloads).

2. Install PyTorch for your computer with the appropriate command. For instance, for training with CPU and a Mac with Python 2.7 (should be default for most)
    ```
    pip install https://s3.amazonaws.com/pytorch/whl/torch-0.1.9.post2-cp27-none-macosx_10_7_x86_64.whl 
    pip install torchvision
    ```
3. You can now run the code in ``mnist.py`` by doing ``python mnist.py``. It has a few parameters which you can find out by ``python mnist.py -h``.

4. It will download the MNIST dataset and train a convolutional neural network on it. You should expect a test error of about 0.6% after 100 epochs using the parameters in the code (learning rate = 0.1).