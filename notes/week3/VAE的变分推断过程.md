# VAE的变分推断过程

1. VAE要做什么：

首先我们要知道生成图像网络都在做什么。生成图像网络想要用一些图像生成另一些图像，但是要求是要是同一类的（比如给一些手写数字生成另一些手写数字）。这时候，我们可以把这些生成的图片看成是一种分布$P(x)$, 然后我们只要sample一个 $x$ 之后代入$P(x)$, 在$P(x)$ 中生成一张图片就可以得到我们想要的东西。

现在的问题是，$P(x)$ 要怎么得到呢？一种很直接的方式就是，$P(x)$ 是一个分布，那么一定是可以被无限个高斯分布 (正态分布) 近似的。也就是说：$P(x) = \int G(z)dz$。

这时候问题又来了，我们要如何得出这些高斯分布呢？这时候我们就想了，既然这里是要无限个高斯分布合在一起，那为什么不从一个 distribution 中 sample 出这些高斯分布呢？于是便想到了说先弄一个标准正态分布，然后用神经网络将这个正态分布上的每一个点都投影到一个 $\mu$ 和 $\sigma$ , 然后用 $\mu$ 和 $\sigma$ 生成一个高斯分布。

也就是说，VAE的目标就是最大化$P(x)$

2. 怎么做：

Then by using VAE, we need to know the possibility curve of the graph; that is, we need to maximize $\displaystyle\sum_{x} P(x)$, or in other words, $\displaystyle\sum_x log(P(x))$.

Therefore, we have to transform this equation to solve this problem. Here, we create a new function $f$ to help us better achieve the goal:
$$
\begin{align*}
log(P(x)) =& \ log(P(x)) \times \displaystyle\int f(z|x)dz \\
=& \ \displaystyle\int f(z|x)\times log(\frac{P(z,x)}{P(z|x)}) dz \\
=& \ \displaystyle\int f(z|x)\times log(\frac{P(z,x)\times f(z|x)}{f(z|x) \times P(z|x)})dz \\
=& \ \displaystyle\int f(z|x)\times log(\frac{P(z,x)}{f(z|x)}) dz \ + \displaystyle\int f(z|x)\times log(\frac{f(z|x)}{P(z|x)}) dz\\
=& \ \displaystyle\int f(z|x)\times log(\frac{P(z,x)}{f(z|x)}) dz \ + KLD(f(z|x)||P(z|x))\\
\end{align*}
$$
In this case, as $KLD(f(z|x)||P(z|x)) \ge 0$,  $log(P(x)) \ge \displaystyle\int f(z|x)\times log(\frac{P(z,x)}{f(z|x)}) dz$.

Also, as $P(x) = \displaystyle\int P(x|z)P(z)dz$, $P(x)$ is independent to $f(x)$, so when optimizing $f(z|x)$ to minimize $KLD(f(z|x)||P(z|x))$, the $log(P(x))$ will gradually become closer to $\displaystyle\int f(z|x)\times log(\frac{P(z,x)}{f(z|x)}) dz$, or in other words, the lower bound of $log(P(x))$.

Therefore, in order to maximize $log(P(x))$, we need to maximize the lower bound $\displaystyle\int f(z|x)\times log(\frac{P(z,x)}{f(z|x)}) dz$.

To maximize the lower bound, we need further processing:
$$
\begin{align*}
Lowerbound =& \displaystyle\int f(z|x)\times log(\frac{P(z,x)}{f(z|x)}) dz\\
=& \displaystyle\int f(z|x)\times log(\frac{P(x|z)\times P(z)}{f(z|x)}) dz\\
=& -\displaystyle\int f(z|x)\times log(\frac{f(z|x)}{P(x|z)}) dz \ + \displaystyle\int f(z|x)\times log(\frac{P(z)}{f(z|x)}) dz\\
=& -KLD(f(z|x)||P(z))\ + E_{f(z|x)}[log(P(x|z))]
\end{align*}
$$
Therefore, $log(P(x)) = KLD(f(z|x)||P(z|x))- KLD(f(z|x)||P(z)) + E_{f(z|x)}[log(P(x|z))]$.

By transforming the equation above, we can get the final equation to compute the loss for the model.
$$
\begin{align*}
log(P(x)) - KLD(f(z|x)||P(z|x)) =& - KLD(f(z|x)||P(z)) + E_{f(z|x)}[log(P(x|z))]\\
=& \ ELBO\\
\end{align*}
$$
And our goal is to construct a neural network to compute a function f to maximize ELBO.

By setting a prior distribution $P(z)$ as a normal distribution $N(0,1)$, we can use a neuro network to generate $\mu$ and $\sigma$ of $f(z|x)$ and maximizing the KLD part of the ELBO by using a Kullback-Leibler divergence $L_{KL} =  - KLD(N(\mu,\sigma),N(0,I))$; Maximizing the reconstruction loss $L_{reconstruct}$ will maximize the rest part of the ELBO.