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
