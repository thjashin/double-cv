# Double Control Variates for Gradient Estimation in Dicrete Latent Variable Models

Code for reproducing the experiments in 
"Double Control Variates for Gradient Estimation in Dicrete Latent Variable Models"
[https://arxiv.org/abs/2111.05300](https://arxiv.org/abs/2111.05300)


## Dependencies
```
tensorflow >= 2.5.0
tensorflow-datasets >= 4.2.0
tensorflow-probability >= 0.12.2
scipy >= 1.6.3
absl >= 0.12.0
pandas >= 1.2.4
numpy >= 1.19.5
tqdm >= 4.60.0
```

## Usage
Toy experiments (quadratic loss):
```
quadratic_loss.ipynb
```

Running VAE experiments:
```
python src/experiment_launcher_singlelayer.py --dataset={dataset} --genmo_lr={lr} --infnet_lr={lr} --encoder_type={net} --grad_type={grad_type} --K={K} --D=200 --seed={seed}
```

- `dataset`: 
  - dynamically binarized: `mnist`, `fashion_mnist`, `omniglot`.
  - non-binarized: `continuous_mnist`, `continuous_fashion`, `continuous_omniglot`.
- `net`: VAE network type, `linear` or `nonlinear`.
- `lr`: 
  - dynamically binarized: `1e-3`.
  - non-binarized: `1e-4`.
- `grad_type`: 
  - REINFORCE leave-one-out: `reinforce_loo`
  - DisARM (Dong et al., 2020): `disarm`
  - RELAX (Grathwohl et al., 2017): `relax` (not affected by `K`, always using 3 evaluations of `f`)
  - ARMS (Dimitriev & Zhou, 2020): `arms`
  - Double CV: `double_cv`
- `K`: number of samples used, equivalent to number of evaluations of `f` in gradient estimators except RELAX.
- `seed`: 1-5. 

## Citation

To cite this work, please use
```
@article{titsias2021double,
  title={Double Control Variates for Gradient Estimation in Dicrete Latent Variable Models}, 
  author={Michalis K. Titsias and Jiaxin Shi},
  journal={arXiv preprint arXiv:2111.05300},
  year={2021}
}
```

## Acknowledgement

The code is based on DisARM (https://github.com/google-research/google-research/tree/master/disarm/binary) and ARMS (https://github.com/alekdimi/arms)
