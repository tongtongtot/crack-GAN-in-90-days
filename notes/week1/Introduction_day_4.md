# GAN的变体之一：CGAN

$CGAN$是$GAN$的一种变体，主要就是加入了 $label$ 来影响生成器生成的图片，以达成一定程度的分类效果。

以 $MNIST$ 数据集为例，$MNIST$ 数据集中有 $0-9$ 共$10$个数字, 所以可以给每一个数据加上一个标签再放入 $generator$ 生成

这样在最后输出的时候就可以通过插入标签来生成指定的图片种类。

比如我们可以通过 $nn.Embedding()$ 函数来实现这个功能

```python
def forward(self, x, labels):
  #...
  self.label_emb = nn.Embedding(10, 10) #embedding 是啥
	x = torch.cat([x, c], 1)
  #...
```

通过这个方法，我们就可以将 $labels$ 的信息插入 $generator$ 和 $discriminator$，实现CGAN的功能。

于是我们生成的时候只需要在原本生成的噪声$Z$ 后面在插入这些 $labels$ 即可。

此处，笔者产生了个疑问：$为啥要在discriminator$里也插入 $labels$ 啊，不应该只需要$generator$插入就行了吗

此处我们需要回归 $generator$ 和 $discriminator$ 的定义。

$generator$ 其实就是将一个分布映射到另一个分布的函数，所以我们做的是将一个随机数输进 $generator$ 产生一张假的图片来交给 $discriminator$ 判断这是来自原图像还是生成的图像。

因此，$discriminator$ 实际上要做的工作就是判断生成的图像是否与原图接近。

那么如果我们在交给 $discriminator$ 判断前，在原图像旁边插上一排 $labels$, 如下图：

![image-20230601230247105](/Users/tongtongtot/Library/Application Support/typora-user-images/image-20230601230247105.png)

并将这个作为 “原图” 输入进 $discriminator$， 与来自 $generator$ 的图片进行比较，并减小差距 ($loss$) ；这时候 $generator$ 就会知道要去生成和这张 "原图" 相近的图片。由于 $generator$ 中下半部分已经确定了 ， 因此最后 $generator$ 只能让上半部分的图片更加接近原图。所以 $generator$ 能够通过这个标签生成类似 $1$ 的图片。