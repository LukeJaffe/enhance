# CS231N class project repo

docker/
- contains stuff to get a docker image running for this stuff (most importantly, pytorch). Image is the one I use for class so it also has tensorflow. Uses python3.

sisr/
- contains working repo for super-resolution with pytorch: https://github.com/pytorch/examples/tree/master/super_resolution
- repo is modified to train on STL10 dataset, 100k 96x96 patches similar to CIFAR10

exp/
- runs several different networks on CIFAR10 in pytorch: https://github.com/kuangliu/pytorch-cifar
- this is copypasted from another thing I was working on, but should be able to train on CIFAR10 with super-resolution as a transform with a few mods
