# SparseLearning-nim
My playground for training sparse neural networks in Nim.

# Summary
This is a simplified explanation of how the algorithms work.

The idea is that a neural networks with dense weights has sparse activations for individual inputs. But this is not commonly utilized, instead dense algorithms are used. But how do we find the largest activations without calculating all activations first? The answer is: hashing! 

Let's start off with how a linear layer works. You have an input row-vector `x` of shape `(1, n_features)` and weights `W` with shape `(n_features, n_outputs)`. The output of the linear layer (assume 0 bias/offset) is `x*W` which will produce an output with shape `(1, n_outputs)`. The key here is to realize how each of the elements in the output is calculated: The `n`:th output is the **dot product** of `x` and the `n`:th column of `W`. It is a simple dot product between two vectors. This will be important in the next step, the hashing.

The idea of the hashing is to construct a vector space such that two vectors which have the same angle are close to each other. This is important because the dot product between two vectors `x*y = |x|*|y|*cos(φ)` is proportional to the angle φ between the two vectors. Assuming that the weight vectors are roughly equal in amplitude, the angle will play a big part in how large the dot product becomes. So by using a vector space where two vectors are close to each other if the angle between them is small, we can reformulate this into a search problem: The columns of `W` which are the closest to `x` in this new vector space are the ones that are *most likely* to produce high activations. So by constructing a hash table which sorts points close to each other into the same bins, the weight vectors can be stored in the hash table. And when we want to calculate the output for an input vector `x`, we hash `x` and calculates the weight vectors that are in the same bin. This way we have will have to do fewer dot products as we can skip many of the weight vectors. Add to that the fact that the output of a layer will be sparse. Hence will the dot product between it and the (dense) weight vector of the next layer require fewer multiplications because most of the elements are zero. 

If we assume the fraction of activations that we choose is `s` and the number of numerical operations of a dense layer is `N^2`. Then we would only need to computed `s*N` of the output nodes, and each dot product would only need `s*N` operations. This hand-wavely gives a final complexity of `s^2*N^2`. This means that if we have a sparsity of `s = 10%`, then only `s^2 = 1%` of the weights will be used in the forward pass. And when doing the backward pass, only these `1%` of the weights will have to be updated. The hashing of the input has complexity `N` so it can be cheap if an efficient enough hashing algorithm is used.

The beauty of this is that it could work with existing pretrained models because it used dense weights. So in theory a library that implements these algorithms could run existing networks faster and on CPUs. 

# References
- [SLIDE : In Defense of Smart Algorithms over Hardware Acceleration for Large-Scale Deep Learning Systems](https://arxiv.org/abs/1903.03129)
- [Scalable and Sustainable Deep Learning via Randomized Hashing](https://arxiv.org/abs/1602.08194)
- [On Symmetric and Asymmetric LSHs for Inner Product Search](https://arxiv.org/abs/1410.5518)
- [BOLT: An Automated Deep Learning Framework for Training and Deploying Large-Scale Neural Networks on Commodity CPU Hardware](https://arxiv.org/abs/2303.17727)
