# BranchNet

An implementation of the paper entitled “Collaborative Learning with A Multi-Branch Framework for Feature Enhancement”.

# Training

```python
python train.py --net_type resnet164 --data_path your_path --dataset cifar100 --gpu_idx 0
```
Note:  -init is default True.  If true, training with the hyper-parameters in lib/args. If set, it is false. You need to set the hyper-parameters via training command.

# Pre Trained Weights

百度云:[BranchNet_ResNet50](https://pan.baidu.com/s/11ee2gRfJQaE_gQVV5TdoQw)  提取码：xq9n 

google driver: [BranchNet_ResNet50](https://drive.google.com/file/d/1iF-wVltMw8jA_bi-MkQJCy6EEyqJiYiy/view?usp=sharing)
