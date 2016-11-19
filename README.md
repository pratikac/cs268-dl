Introductory lecture on deep learning in CS 268 at UCLA.

Example code in torch added.

Install Torch from [http://torch.ch/docs/getting-started.html](http://torch.ch/docs/getting-started.html).

In addition to this, you will need two packages:
- Install the "mnist" package using `luarocks install mnist`
- Install the fakecuda package from [https://github.com/soumith/fakecuda](https://github.com/soumith/fakecuda)

Run the code as
```
th mnist.lua -h
```

This is LeNet-style network for MNIST, it should train to about ~1% testing error in 10 epochs and should get to ~0.85% test error around 25 epochs.
