# SAFFNet: self-attention based on Fourier frequency domain filter network for visual question answering


### Dataset setup
see [DATA.md](DATA.md)

### Config Introduction
In [trar.yml](configs/vqa/trar.yml) config we have these specific settings for `TRAR` model
```
ORDERS: [0, 1, 2, 3]
IMG_SCALE: 8 
ROUTING: 'hard' # {'soft', 'hard'}
POOLING: 'attention' # {'attention', 'avg', 'fc'}
TAU_POLICY: 1 # {0: 'SLOW', 1: 'FAST', 2: 'FINETUNE'}
TAU_MAX: 10
TAU_MIN: 0.1
BINARIZE: False
```
- `ORDERS=list`, to set the local attention window size for routing.`0` for global attention.
- `IMG_SCALE=int`, which should be equal to the `image feature size` used for training. You should set `IMG_SCALE: 16` for `16 × 16` training features.
- `ROUTING={'hard', 'soft'}`, to set the `Routing Block Type` in TRAR model.
- `POOLING={'attention', 'avg', 'fc}`, to set the `Downsample Strategy` used in `Routing Block`.
- `TAU_POLICY={0, 1, 2}`, to set the `temperature schedule` in training TRAR when using `ROUTING: 'hard'`.
- `TAU_MAX=float`, to set the maximum temperature in training.
- `TAU_MIN=float`, to set the minimum temperature in training.
- `BINARIZE=bool`, binarize the predicted alphas (alphas: the prob of choosing one path), which means **during test time**, we only keep the maximum alpha and set others to zero. If `BINARIZE=False`, it will keep all of the alphas and get a weight sum of different routing predict result by alphas. **It won't influence the training time, just a small difference during test time**.

**Note that please set `BINARIZE=False` when `ROUTING='soft'`, it's no need to binarize the path prob in soft routing block.**

**`TAU_POLICY` visualization**

For `MAX_EPOCH=13` with `WARMUP_EPOCH=3` we have the following policy strategy:
<p align="center">
	<img src="misc/policy_visualization.png" width="550">
</p>

### Training
**Train model on VQA-v2 with default hyperparameters:**
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar'
```
and the training log will be seved to:
```
results/log/log_run_<VERSION>.txt
```
Args:
- `--DATASET={'vqa', 'clevr'}` to choose the task for training
- `--GPU=str`, e.g. `--GPU='2'` to train model on specific GPU device.
- `--SPLIT={'train', 'train+val', train+val+vg'}`, which combines different training datasets. The default training split is `train`.
- `--MAX_EPOCH=int` to set the total training epoch number.


**Resume Training**

Resume training from specific saved model weights
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --RESUME=True --CKPT_V=str --CKPT_E=int
```
- `--CKPT_V=str`: the specific checkpoint version
- `--CKPT_E=int`: the resumed epoch number

**Multi-GPU Training and Gradient Accumulation**
1. Multi-GPU Training:
Add `--GPU='0, 1, 2, 3...'` after the training scripts.
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --GPU='0,1,2,3'
```
The batch size on each GPU will be divided into `BATCH_SIZE/GPUs` automatically.

2. Gradient Accumulation:
Add `--ACCU=n` after the training scripts
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --ACCU=2
```
This makes the optimizer accumulate gradients for `n` mini-batches and update the model weights once. `BATCH_SIZE` should be divided by `n`.

### Validation and Testing
**Warning**: The args `--MODEL` and `--DATASET` should be set to the same values as those in the training stage.

**Validate on Local Machine**
Offline evaluation only support the evaluations on the `coco_2014_val` dataset now.
1. Use saved checkpoint
```bash
python3 run.py --RUN='val' --MODEL='trar' --DATASET='{vqa, clevr}' --CKPT_V=str --CKPT_E=int
```

2. Use the absolute path
```bash
python3 run.py --RUN='val' --MODEL='trar' --DATASET='{vqa, clevr}' --CKPT_PATH=str
```

**Online Testing**
All the evaluations on the `test` dataset of VQA-v2 and CLEVR benchmarks can be achieved as follows:
```bash
python3 run.py --RUN='test' --MODEL='trar' --DATASET='{vqa, clevr}' --CKPT_V=str --CKPT_E=int
```

Result file are saved at:

`results/result_test/result_run_<CKPT_V>_<CKPT_E>.json`

You can upload the obtained result json file to [Eval AI](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview) to evaluate the scores.

### Models
Here we provide our pretrained model and log, please see [MODEL.md](MODEL.md)

## Acknowledgements
- [openvqa](https://github.com/MILVLG/openvqa)
- [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa)

