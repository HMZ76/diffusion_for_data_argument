## 使用说明

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

### classifer_free_guidance训练
``` 
##单卡训练

cd classifer_free_guidance
python classifer_free_guidance/single_cfg_train.py
```

```
##多卡训练

cd classifer_free_guidance
python -m torch.distributed.launch --nproc_per_node=8 multi_cfg_train.py

```
### dps训练
```
cd dps
#训练
python score_matching.py
#dps采样
python score_matching.py --sample --model_path ./checkpoints_ddpm_score/best_model.pt
```

### eval
