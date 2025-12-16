## 使用说明

### 配环境
```
conda create -n tpdf python=3.10

pip install -r requirements.txt

```

### 下载数据集 
```
wget https://sven-mayer.com/datasets/2022-MobileHCI-SuperRes/16h5_training.pkl

wget https://sven-mayer.com/datasets/2022-MobileHCI-SuperRes/16h5_validation.pkl

wget https://sven-mayer.com/datasets/2022-MobileHCI-SuperRes/36h11_training.pkl

wget https://sven-mayer.com/datasets/2022-MobileHCI-SuperRes/36h11_validation.pkl
```

### 数据预处理
```
mkdir ./data
mv *.pkl ./data
python preprocess_data.py
```

### classifer_free_guidance

##### 训练
``` 
##单卡训练

cd classifer_free_guidance
python single_cfg_train.py
```

```
##多卡训练

cd classifer_free_guidance
python -m torch.distributed.launch --nproc_per_node=8 multi_cfg_train.py

```

##### 采样
```
python sample.py \
  --model_path ./checkpoints/multiscale_model_epoch_8.pt \
  --num_samples 10000 \
  --cfg_scale 2.5 \
  --sample_steps 60 \
  --output_root ./results \
  --batch_size 512

```

### Diffusion Posterior Sampling

##### 训练
```
cd dps
#训练
python score_matching_train.py
```

##### 采样
```
python sample.py \
  --model_path ./checkpoints_ddpm_score/best_model.pt \
  --num_samples 20 \
  --cond_data_root ../val_data/ \
  --output_dir ./results \
  --device cuda
```


### eval(FID似乎有点大？)
```
python eval.py --gen_folder ./results/generated_images --target_folder classifer_free_guidance/results/target_images

#LPIPS: 0.0547
#FID: 22.5981

```

