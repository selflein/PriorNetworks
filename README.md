# PriorNetworks

## Updates for this fork
This fork tries to reproduce the experiments on the synthetic dataset in Section 5.1 of the [PriorNetworks paper](https://arxiv.org/abs/1802.10501).

### Experiment reproduction results

<p align="center">
<img src="./imgs/results.png" width="80%">
</p>

---

### Setup
Install conda environment 
```
conda create -f environment.yml
```

Train on the Gaussian synthetic dataset
```
python prior_networks/priornet/run/train_dpn_synth.py --checkpoint_path checkpoints/dpn_synth_std4 --model_dir checkpoints/dpn_synth_std4 --lrc 10 --lrc 30 --n_epochs 1000 --lr 1e-4 --gpu 0 --std 4
```
the values of {std} (which is the variance of the Gaussians in the dataset) in the paper are 1 and 4. 

Alternatively, pretrained models can be found [here](https://drive.google.com/drive/folders/1c2JpANMImZhqxc29cCn-mlmz3B3arDok?usp=sharing). Place the `checkpoint` folder in the root of the repo.

### Generate figures 
Run the `reproduce_gaussian_exp.ipynb` notebook to generate the plots found above.
