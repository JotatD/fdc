# Installation

Create conda environment (also for cluster):
```bash
conda env create -f environment.yaml
```
Activate environment:
```bash
conda activate maxent
```
In root directory, install package (maybe not necessary, just in case):
```bash
pip3 install -e .
```

# Running stuff locally from configs

Default config path is specified in the script:

```bash
python scripts/toy_fdc.py
```

# Running grid search (generic example)


## Locally


```bash
python your_script.py -m hydra/launcher=joblib hydra.launcher.n_jobs=4 param1=1,2 param2=0.1,0.2
```

## On Slurm (euler)

In general, store models on cluster in 

```bash
/cluster/project/krause/$USER/models/
```

Concrete example (should just run out of the box if model paths are configured properly):
```bash
python scripts/fdc_toy.py -m hydra/launcher=slurm trainer.lr=1.e-5,1.e-4 exp_dir=/cluster/project/krause/$USER/fdc_neurips2025_test model_path=/cluster/project/krause/$USER/models/model_single_gauss.pth
```

If there are space issues, might be smart to store stuff on SCRATCH:
```bash
python scripts/fdc_toy.py -m hydra/launcher=slurm trainer.lr=1.e-5,1.e-4 exp_dir=$SCRATCH/fdc_neurips2025_test model_path=/cluster/project/krause/$USER/models/model_single_gauss.pth
```

/cluster/project/krause/mvlastelica/models/model_single_gauss.pth


## On Node (joblib)

```bash
python   scripts/stable_diffusion_exp_fdc.py -m hydra/launcher=joblib trainer.alpha_div=0.0,1.0,10.0 trainer.epsilon=5.e-4 trainer.lmbda=0.01,0.05
```


# Stable diffusion runs

For these you need to first download the SD-1.5 model checkpointe via huggingface cli, then set HF_HOME appropriately for local and cluster runs.

## Single diffusion run

```bash
 # the free cuda devices currently
export CUDA_VISIBLE_DEVICES=0 && python   scripts/stable_diffusion_exp_fdc.py --config-name=sd_fdc_single trainer.alpha_div=20.0
```


## Diffusion run on cluster

```bash
python   scripts/stable_diffusion_exp_fdc.py hydra/launcher=slurm_a100_40g  trainer.alpha_div=0.0,10.0,100.0 trainer.lmbda=0.5  +trainer.epsilon=5.e-4 --config-name=sd_fdc_slurm -m
```

## LIST OF SCRIPTS FOR FUNCTIONALS:

Entropy-KL script: 
```bash
python scripts/entropy_KL.py -m hydra/launcher=slurm trainer.lr=3.e-4,5.e-4 exp_dir=/cluster/project/krause/$USER/fdc_neurips2025_test model_path=/cluster/project/krause/$USER/models/unbalanced_ellipses.pth
```

For single run:
```bash
python scripts/entropy_KL.py trainer.lr=3.e-4 exp_dir=/tmp/$USER/fdc_neurips2025_test model_path=/home/$USER/projects/max-entropy-diff/models/unbalanced_ellipses.pth wandb.entity=mvlast ++output_dir=/tmp/test
```

## LIST OF GRID SEARCHES

Entropy-KL script:
```bash
python scripts/entropy_KL.py -m hydra/launcher=slurm ++trainer.alpha_div=0.0,10.0,50.0 trainer.lr=1.e-4,3.e-4 trainer.lmbda=1.0,2.0,4.0 trainer.finetune_steps=2,10,100 loop.md_steps=50 loop.training_steps=500 model_path=/cluster/project/krause/$USER/models/unbalanced_ellipses.pth wandb.entity=mvlast
```

Wasserstein:
```bash
python scripts/W1.py -m hydra/launcher=slurm trainer.lr=3.e-4,5.e-4 trainer.lmbda=0.5,1.0,4.0,10.0,50.0 trainer.finetune_steps=100 loop.md_steps=10 loop.training_steps=500 trainer.critic_lr=1.e-3 ++trainer.alpha_div=50,100,1000.0 model_path=/cluster/project/krause/$USER/models/model_gaussian_circle.pth wandb.entity=mvlast 'wandb.tags=[W1, W1-v1]'
```

test
```bash
python scripts/W1.py -m hydra/launcher=slurm trainer.lr=3.e-4 trainer.lmbda=10.0 trainer.finetune_steps=100 loop.md_steps=4 trainer.critic_lr=5.e-3  model_path=/cluster/project/krause/$USER/models/model_gaussian_circle.pth wandb.entity=mvlast
```