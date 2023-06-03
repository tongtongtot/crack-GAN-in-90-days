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

### 7. 保存图片

```python
from torchvision.utils import save_image
save_image(picture, path, nrow = 8, Normalize = True)
```

