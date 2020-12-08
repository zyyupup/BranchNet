# BranchNet

An implementation of the paper entitled “Collaborative Learning with A Multi-Branch Framework for Feature Enhancement”.

# Training

```python
python train.py --net_type resnet164 --data_path your_path --dataset cifar100 --gpu_idx 0
```
Note:  -init is default True.  If true, training with the hyper-parameters in lib/args. If set, it is false. You need to set the hyper-parameters via training command.
