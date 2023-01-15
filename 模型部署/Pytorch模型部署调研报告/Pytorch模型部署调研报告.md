# Pytorch模型部署调研报告

## 1 模型部署目的
机器学习模型作为一个子模块部署在完整的软件中的完成特定的任务。在缺陷检查的项目中，模型可以部署在生产线上进行缺陷检测、在标注工具中进行辅助标注。结合训练任务，可以及时部署测试模型。

## 2 模型部署方案
### 2.1 原生Web框架
原生Web框架包括Django、Flask等。本节将介绍如何使用Flask搭建一个基于PyTorch的图片分类服务以及并行处理的相关技术。该方案方便对服务化的模型进行debug，因为web开发的同时常常很难定位到深度学习服务的bug的位置。

1. 环境
- 系统：Ubuntu 18.04
- Python版本：3.7

依赖Python包：
PyTorch==1.3
Flask==0.12
Gunicorn
需要注意的是Flask 0.12中默认的单进程单线程，而最新的1.0.2则不是（具体是多线程还是多进程尚待考证），而中文博客里面能查到的资料基本都在说Flask默认单进程单线程。

依赖工具

nginx
apache2-utils
nginx 用于代理转发和负载均衡
apache2-utils用于测试接口

2. 搭建异步服务
对于做算法的读者，不着急搭建深度学习模型，因为算法工程师普遍对web开发不太熟悉，可以先搭建一个最简单的web服务，并验证其功能无误之后再加入深度学习模型。

2.1 Flask搭建图片上传服务
因为图片分类服务需要从本地上传图片，可以先搭建一个用于图片上传的服务
```python
# sim_server.py
from flask import Flask, request
from werkzeug.utils import secure_filename
import uuid
from PIL import Image
import os
import time

app = Flask(__name__)

@app.route("/run",methods = ["GET"])
def run():
    # 用于测试服务是否并行
    time.sleep(1)
    return "0"

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5555,debug=True)
```
启动服务：
```python
python sim_server.py
```

此时可以使用apache-utils测试接口是否是异步运行。如果是单进程单线程的话，每秒钟只能处理一个请求，服务的处理能力会随着进程数的增加而增加，但是由于计算机性能限制，增加进程数带来的处理能力提升会越来越小。

使用gunicorn启动多个进程。
```shell
gunicorn -w 4 -b 0.0.0.0:5555 sim_server:app
```

输出如下内容代表服务创建成功：
```shell
[2020-02-11 14:50:24 +0800] [892] [INFO] Starting gunicorn 20.0.4
[2020-02-11 14:50:24 +0800] [892] [INFO] Listening at: http://0.0.0.0:5555 (892)
[2020-02-11 14:50:24 +0800] [892] [INFO] Using worker: sync
[2020-02-11 14:50:24 +0800] [895] [INFO] Booting worker with pid: 895
[2020-02-11 14:50:24 +0800] [896] [INFO] Booting worker with pid: 896
[2020-02-11 14:50:24 +0800] [898] [INFO] Booting worker with pid: 898
[2020-02-11 14:50:24 +0800] [899] [INFO] Booting worker with pid: 899
```
再次使用apache-utils进行测试，可以看到处理能力的提升：
```shell
ab -c 4 -n 10 http://localhost:5555/run
```
得到处理能力：Requests per second: 3.33 [#/sec] (mean)

1. 将PyTorch分类模型接入服务
```python
from flask import Flask, request
from werkzeug.utils import secure_filename
import uuid
from PIL import Image
import os
import time
import base64
import json

import torch
from torchvision.models import resnet18
from torchvision.transforms import ToTensor

from keys import key

app = Flask(__name__)
net = resnet18(pretrained=True)
net.eval()

@app.route("/",methods=["GET"])
def show():
    return "classifier api"

@app.route("/run",methods = ["GET","POST"])
def run():
    file = request.files['file']
    base_path = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(base_path, "temp")):
        os.makedirs(os.path.join(base_path, "temp"))
    file_name = uuid.uuid4().hex
    upload_path = os.path.join(base_path, "temp", file_name)
    file.save(upload_path)

    img = Image.open(upload_path)
    img_tensor = ToTensor()(img).unsqueeze(0)
    out = net(img_tensor)
    pred = torch.argmax(out,dim = 1)

    return "result : {}".format(key[pred])

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5555,debug=True)
```
并发测试
使用apache2-utils进行上传图片的post请求方法参考：

https://gist.github.com/chiller/dec373004894e9c9bb38ac647c7ccfa8

严格参照，注意一个标点，一个符号都不要错。

使用这种方法传输图片的base64编码，在服务端不需要解码也能使用

然后使用下面的方式访问

gunicorn 接口

ab -n 2 -c 2 -T "multipart/form-data; boundary=1234567890" -p turtle.txt http://localhost:5555/run
nginx 接口

ab -n 2 -c 2 -T "multipart/form-data; boundary=1234567890" -p turtle.txt http://localhost:5556/run
有了gunicorn和nginx就可以轻松地实现PyTorch模型的多机多卡部署了


### 2.2 无服务框架

### 2.3 机器学习专用框架

## 3 总结