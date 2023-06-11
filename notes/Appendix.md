# Appendix

### 1. Git 使用方法

```shell
git pull
git add *
git commit -m "message"
git push
git pull
```

### 2. parser

```python
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--something", type = some_type, default = default, help = "something")
opt = parser.parse_args()
```

当然，也可以在$options.py$ 文件中写好，然后在外面调用：

```python
class options():
		def get_opt():
				parser = argparse.ArgumentParser()
				parser.add_argument("")
				opt = parser.parse_args()
				return opt
		'''
		这之后还可以def其他的函数来实现别的功能
		比如.cuda就可以在这里面实现
		...
```

### 3. 生成文件路径

```python
import os
os.makedirs(path, exist_ok = True)
```

### 4. 保存模型

```python
import torch
torch.save(model.state_dict(), path)
```

### 5. 加载模型

```python
import torch
model.load_state_dict(torch.load(path))
```

### 6. 并行GPU

```python
import torch.nn as nn
gpus = [0,1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model.to(device), device_ids = gpus, output_device = gpus[0])
```

注：在使用之后 $torch.save$ 的话可能会出现在前面多一个 $module.$ 的情况

可以使用以下方式解决：（provided by chatGPT）

```python
from collections import OrderedDict
state_dict = torch.load(path)
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k[7:]  # remove 'module.' prefix
    new_state_dict[name] = v

# load params
model.load_state_dict(new_state_dict)
```

另外要记得如果要用 $GPU$ 的话那所有的 $Tensor$ 都要 $.to(device)$

注：有时候会出现**AttributeError: ‘DataParallel’ object has no attribute **的报错信息，

这时候可以在调用model.函数的时候在model.后面加个module. 

如下：

```python
# model.set_input() -> model.module.set_input()
```

### 7. 保存图片

```python
from torchvision.utils import save_image
save_image(picture, path, nrow = 8, Normalize = True)
```

### 8. Tqdm 进度条

首先是 import 库，这个很重要，而且很容易报错

```python
from tqdm import tqdm
# 一定要这样import, 直接 import tqdm 会报错
```

然后$tqdm$有两种实现方式，一种是直接套在 $for$ 循环外面，另一种是在 $for$ 循环外用 $with$ 调用。

具体实现方式如下：

```python
for epoch_idx in tqdm(range(opt.epoch), desc="进度条左边的文字"):
```

```python
with tqdm(total=opt.epoch) as t:
		for epoch_idx in range(opt.epoch):
				t.set_description("进度条左边文字")
				t.set_postfix(loss = model.get_loss())
				# 这就会显示： loss = ...
				sleep(0.1)
				t.update(1)
```

### 9. 调loss的小技巧

当有两个$loss$的时候要让两个$loss$在接近的范围内

比如：

当有$L1Loss$ 和 $KLD$ 这两个 $loss$的时候：

​	$L1Loss = 0.01$

​	$KLD = 100$

那就可以考虑在$KLD$ 前加一个$10^{-4}$ 的系数。
