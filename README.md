# Text Supervised LFM

![arch](assets/arch.png)

This contains the implementation for local feature mixup as well as the training procedure.

Example usage:
```
python main.py --cfg config/general.yaml config/imagenet.yaml config/proposed/lfm-mms.yaml
```

## LFM

- Local Sampling is defined as a PyTorch Sampler called LocalClassSampler, which is located in `data/dataloader.py`.
- All Label shift and other mixup procedures are defined inside `mixups.py`.


## Configuration

You may specify configuration through one or more yaml files with the `--cfg` flag. If multiple yaml files are specified like the following example, then they are merged together to form one configuration. Note that the `--cfg` flag only needs to be specified once.

```
python main.py --cfg config/general.yaml config/proposed/lfm-mms.yaml
```

If multiple yaml files contain a definition for a field, then the last-mentioned yaml file in the command line args overwrites all preceding specifications of that particular field. Observing the above example, `use_lfm` will be set to true.

Current list of configuration fields:
```
loss: string, must be one of: "CE", "BalCE", "Focal", "LDAM", "MMS"
dataset: string, must be one of: "CIFAR100", "CIFAR10", "ImageNet", "Places", "iNaturalist18"
cifar_imb: int, imbalance ratio if a CIFAR dataset is specified
epochs: [int, int], epochs for phase 0 and 1
lr: [float, float], learning rates for phase 0 and 1
use_lfm: bool, whether to use lfm or not
alpha: float, label shift intensity
tau: float, local sample intensity (lower is more)
batch_size: int
backbone: string, must be one of: "RN50", "RN50x16", "RN101", "ViT-B/32", "ViT-B/16"
```

## CLI Args
```
--cfg: string, specifies one or more configuration files (see above)
--gpu: int, if on an environment with multiple gpus, but you want to use a single gpu, then use this to specify which gpu you want to train on
--checkpoint: string, specify a checkpoint to continue phase 1 training
```
