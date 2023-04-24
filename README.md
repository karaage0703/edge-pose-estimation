# edge-pose-estimation
AI pose estimation with edge device

# Setup
## Hardware

- Raspberry Pi 4 or Raspberry Pi 3(Raspberry Pi 4 is recommended)
- USB Web camera

## Requirements
- onnxruntime==1.12.1
- opencv-python==4.6.0.66

## Download this repository
Execute following command:

```sh
$ cd && git clone https://github.com/karaage0703/edge-pose-estimation
```

# Usage
## Pose estimation with raspi cam module
Execute following commands:

```sh
$ cd ~/edge-pose-estimation
$ python3 demo_onnx.py
```

# License
This software is released under Apache-2.0 license, see LICENSE.

# Author
- [@karaage0703](http://github.com/karaage0703)

# Special Thanks
- [Kazuhito00](https://github.com/Kazuhito00)
- [PINTO0309](https://github.com/PINTO0309)

# References
- [Kazuhito00/MoveNet-Python-Example](https://github.com/Kazuhito00/MoveNet-Python-Example) 
- [PINTO_model_zoo/115_MoveNet](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/115_MoveNet)
- [Kazuhito00/PyCon-mini-Shizuoka-2021-Talk4](https://github.com/Kazuhito00/PyCon-mini-Shizuoka-2021-Talk4)
