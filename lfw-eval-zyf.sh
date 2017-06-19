### 1
# align faces using MTCNN
export PYTHONPATH=/disk2/zhaoyafei/facenet-tensorflow/src
for N in {1..3}; do python src/align/align_dataset_mtcnn.py /disk2/data/FACE/LFW/LFW /disk2/data/FACE/LFW/lfw_mtcnnpy_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25 & done
###

### 2.1 test
# run LFW test with the first model
python src/validate_on_lfw.py /disk2/data/FACE/LFW/lfw_mtcnnpy_160 ./20170511-185253
###

### 2.1results
# I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties:
# name: Tesla K80
# major: 3 minor: 7 memoryClockRate (GHz) 0.8235
# pciBusID 0000:06:00.0
# Total memory: 11.17GiB
# Free memory: 11.11GiB
# I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0
# I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y
# I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:06:00.0)
# Model directory: ./20170511-185253
# Metagraph file: model-20170511-185253.meta
# Checkpoint file: model-20170511-185253.ckpt-80000
# Runnning forward pass on LFW images
# Accuracy: 0.988+-0.005
# Validation rate: 0.92500+-0.02156 @ FAR=0.00100
# Area Under Curve (AUC): 0.999
# Equal Error Rate (EER): 0.014
###

### 2.2 test
# run LFW test with the second model
python src/validate_on_lfw.py /disk2/data/FACE/LFW/lfw_mtcnnpy_160 ./20170512-110547
###

### 2.2 results
# I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties:
# name: Tesla K80
# major: 3 minor: 7 memoryClockRate (GHz) 0.8235
# pciBusID 0000:06:00.0
# Total memory: 11.17GiB
# Free memory: 11.11GiB
# I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0
# I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y
# I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:06:00.0)
# Model directory: ./20170512-110547
# Metagraph file: model-20170512-110547.meta
# Checkpoint file: model-20170512-110547.ckpt-250000
# Runnning forward pass on LFW images
# Accuracy: 0.992+-0.003
# Validation rate: 0.97467+-0.01477 @ FAR=0.00133
# Area Under Curve (AUC): 1.000
# Equal Error Rate (EER): 0.007
###