### TVM增加车型识别模型的编译支持

* 模型种类: [peleenet](https://arxiv.org/abs/1804.06882v1)

资源文件说明:

* assets/mxnet_peleenet_v4_nopad.pdf:模型可视化结构图
* assets/Pelee: A Real-Time Object Detection System on Mobile Devices.pdf:模型原始论文

#### 修改 topi/python/topi/nn/conv2d.py

_WORKLOADS列表增加如下行

```python
    # workloads of pelee net on mmr
    # stem 
    Workload('float32', 'float32', 224, 224, 3, 32, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 112, 112, 32, 16, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 112, 112, 16, 32, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 56, 56, 64, 32, 1, 1, 0, 0, 1, 1),
    # stage_1
    Workload('float32', 'float32', 56, 56, 32, 16, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 16, 32, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 56, 56, 64, 16, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 96, 16, 1, 1, 0, 0, 1, 1),
    # stage_1_transition
    Workload('float32', 'float32', 56, 56, 128, 128, 1, 1, 0, 0, 1, 1),
    # stage_2
    Workload('float32', 'float32', 28, 28, 128, 32, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 32, 32, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 28, 28, 160, 32, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 192, 32, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 224, 32, 1, 1, 0, 0, 1, 1),
    # stage_2_transition
    Workload('float32', 'float32', 28, 28, 256, 256, 1, 1, 0, 0, 1, 1),
    # stage_3
    Workload('float32', 'float32', 14, 14, 256, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 64, 32, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 14, 14, 288, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 320, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 352, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 384, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 416, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 448, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 480, 64, 1, 1, 0, 0, 1, 1),
    # stage_3_transition
    Workload('float32', 'float32', 14, 14, 512, 512, 1, 1, 0, 0, 1, 1),
    # stage_4
    Workload('float32', 'float32', 7, 7, 512, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 64, 32, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 7, 7, 544, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 576, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 608, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 640, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 672, 64, 1, 1, 0, 0, 1, 1),
    # stage_4_transition
    Workload('float32', 'float32', 7, 7, 704, 704, 1, 1, 0, 0, 1, 1),
```

#### 修改 topi/python/topi/rasp/conv2d.py

_SCHEDULES列表增加如下行

```python
    # pelee net
    # stem 
    SpatialPack(2, 2, 4, 28, 1, True),
    SpatialPack(1, 4, 8, 14, 1, False),
    SpatialPack(1, 4, 4, 1, 4, False),
    SpatialPack(1, 4, 8, 4, 1, True),
    # stage_1
    SpatialPack(1, 4, 4, 4, 1, True),
    SpatialPack(2, 2, 4, 1, 1, True),
    SpatialPack(1, 4, 4, 4, 1, True),
    SpatialPack(1, 4, 4, 4, 1, True),
    # stage_1_transition
    SpatialPack(1, 4, 8, 8, 8, True),
    # stage_2
    SpatialPack(2, 2, 8, 4, 1, True),
    SpatialPack(2, 2, 4, 1, 1, True),
    SpatialPack(2, 2, 8, 4, 1, True),
    SpatialPack(2, 2, 8, 4, 1, True),
    SpatialPack(2, 2, 8, 4, 1, True),
    # stage_2_transition
    SpatialPack(2, 2, 8, 4, 8, False),
    # stage_3
    SpatialPack(2, 2, 8, 4, 4, True),
    SpatialPack(2, 2, 4, 1, 1, True),
    SpatialPack(2, 2, 8, 4, 4, True),
    SpatialPack(2, 2, 8, 4, 4, True),
    SpatialPack(2, 2, 8, 4, 4, True),
    SpatialPack(2, 2, 8, 4, 4, True),
    SpatialPack(2, 2, 8, 4, 4, True),
    SpatialPack(2, 2, 8, 4, 4, True),
    SpatialPack(2, 2, 8, 4, 4, True),
    # stage_3_transition
    SpatialPack(2, 2, 8, 1, 8, False),
    # stage_4
    Im2ColPack(7, 4, 1, 16, False),
    Im2ColPack(7, 4, 1, 16, False),
    Im2ColPack(7, 4, 1, 16, False),
    Im2ColPack(7, 4, 1, 16, False),
    Im2ColPack(7, 4, 1, 16, False),
    Im2ColPack(7, 4, 1, 16, False),
    Im2ColPack(7, 4, 1, 16, False),
    # stage_4_transition
    Im2ColPack(7, 4, 1, 4, True),
```

> _SCHEDULES和_WORKLOADS列表元素的位置必须一一对应