# Score-Based Model
A score-based model repo. Images can be trained `conditionally` or `unconditionally`. 

## 1. Running Experiments

### 1.1. Dependencies

Run the following to install all necessary python packages for our code.

```bash
pip install -r requirements.txt
```

### 1.2. Project structure

`main.py` is the file that you should run for both training and sampling. Execute ```python main.py --help``` to get its usage description:

```
usage: main.py [-h] --config CONFIG [--seed SEED] [--exp EXP] --doc DOC
               [--comment COMMENT] [--verbose VERBOSE] [--test] [--sample]
               [--fast_fid] [--resume_training] [-i IMAGE_FOLDER] [--ni]

optional arguments:
  -h, --help             show this help message and exit
  --config CONFIG        Path to the config file
  --seed SEED            Random seed
  --exp EXP              Path for saving running related data.
  --doc DOC              A string for documentation purpose. Will be the name
                         of the log folder.
  --comment COMMENT      A string for experiment comment
  --conditional          Whether to train a conditional score-based model
  --verbose VERBOSE      Verbose level: info | debug | warning | critical
  --sample               Whether to produce samples from the model after training
  --resume_training      Whether to resume training
  -i, --sampling_folder  Folder to store sampled image after training (default: image_samples)
                         The folder name of samples
  --ni                   No interaction. Suitable for Slurm Job launcher
```

Configuration files are in `config/`. You don't need to include the prefix `config/` when specifying  `--config` . All files generated when running the code is under the directory specified by `--exp`. They are structured as:

```bash
<exp> # a folder named by the argument `--exp` given to main.py
├── <logs> # contains checkpoints and samples produced during training
│   └── <doc> # a folder named by the argument `--doc` specified to main.py
│      └── <conditional> # a folder for logging all conditional results
│         └── <image_samples> # a folder for all samples produced after training
│         ├── <samples>    # a folder for all samples produced during training
│         ├── checkpoint_x.pth # the checkpoint file saved at the x-th training iteration
│         ├── checkpoint.pth   # the checkpoint file saved at the last training iteration
│         ├── config.yml # the configuration file for training this model
│         └── stdout.txt # all outputs to the console during training
│      └── <unconditional> # a folder for logging all unconditional results
│         └── <image_samples> # a folder for all generated samples produced after training
│         ├── <samples> # all samples produced during training
│         ├── checkpoint_x.pth # the checkpoint file saved at the x-th training iteration
│         ├── checkpoint.pth   # the checkpoint file saved at the last training iteration
│         ├── config.yml # the configuration file for training this model
│         └── stdout.txt # all outputs to the console during training    
└── <tensorboard> # tensorboard files for monitoring training
    └── <doc> # this is the log_dir of tensorboard
```

### 1.3. Training

For example, we can train an unconditional 2D abdonemal CT scan

```bash
python main.py --config abdomen2dCT.yml --doc abdomen2dCT
```

Log files will be saved in `<exp>/unconditional/logs/abdomen2dCT`

For conditional training

```bash
python main.py --config abdomen2dCT.yml --doc abdomen2dCT --conditional
```

Log files will be saved in `<exp>/conditional/logs/abdomen2dCT`

## 2. Sampling Experiment

### 2.1. Sampling Images

If we want to sample an image of an unconditional abdonemal CT after training, we can edit `abdomen2dCT.yml` to specify the `ckpt_id` under the group `sampling` or ignore to use the last saved model, and then run the following

```bash
python main.py --doc abdomen2dCT --abdomen2dCT brain.yml --sample
```
For conditional sampling

```bash
python main.py --doc abdomen2dCT --abdomen2dCT brain.yml --sample --conditional
```

Samples will be saved in 

`<exp>/<unconditional>/abdomen2dCT/image_samples` for unconditional.

`<exp>/<conditional>/abdomen2dCT/image_samples` for conditional.

NB: You should have trained a conditional model to sample conditionally or trained unconditional model to sample unconditionally.


### 2.2. Result

Resolution (384 x 384)
![image_gen](https://github.com/SWAG-SwarmLearningforGenerativeModels/Score-Based-Model/assets/77448406/1afc4cf4-bc5c-4667-91fe-de743cc1fe37)

Resolution (512 x 512)
![imgonline-com-ua-twotoone-LaUkV5BQCJ](https://github.com/SWAG-SwarmLearningforGenerativeModels/Score-Based-Model/assets/77448406/f34c0e64-8251-48fd-8c8a-8f6c3fe4c2ce)

## References

```bib
@inproceedings{song2020improved,
  author    = {Yang Song and Stefano Ermon},
  editor    = {Hugo Larochelle and
               Marc'Aurelio Ranzato and
               Raia Hadsell and
               Maria{-}Florina Balcan and
               Hsuan{-}Tien Lin},
  title     = {Improved Techniques for Training Score-Based Generative Models},
  booktitle = {Advances in Neural Information Processing Systems 33: Annual Conference
               on Neural Information Processing Systems 2020, NeurIPS 2020, December
               6-12, 2020, virtual},
  year      = {2020}
}
```

and/or our previous work

```bib
@inproceedings{song2019generative,
  title={Generative Modeling by Estimating Gradients of the Data Distribution},
  author={Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11895--11907},
  year={2019}
}
```