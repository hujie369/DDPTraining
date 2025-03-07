# A DDP training framework
If use single GPU, please use the following command:
```
export model=resnet50
train.py --model=${model} --batch-size=256 --lr0=0.1 --workers=8 --name=${model}_
```

If use multiple GPUs, please use the following command:
```
export OMP_NUM_THREADS=4
export model=resnet50
torchrun --nproc_per_node 4 train.py --model=${model} --batch-size=1024 --lr0=0.1 --workers=8 --name=${model}_ --device=0,1,2,3
```
