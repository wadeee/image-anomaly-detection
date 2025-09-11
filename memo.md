# Memo

## Dataset

You can download the full dataset from the link

https://drive.google.com/file/d/1YgS897POEal8mkP3kngwcV5LI9vuyPuJ/view?usp=sharing

## Installation

Build the environment based on python3.10
```bash
conda create -n image-anomaly-detection python=3.10
conda activate image-anomaly-detection
```

install all the packages using

```bash
pip install -r requirements-custom-gpu.txt
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

if CUDA is not present, then comment torch==2.1.2+cu121 and torchvision==0.16.2+cu121 and use torch==2.1.2 and torchvision==0.16.2. Then install Anomalib
```bash
# pip install anomalib==1.2.0
anomalib install
```

## Train image models

### Patchcore

```bash
python train_anomalib/train_patchcore_anomalib.py --dataset_root C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/dataset/images_lego_256/two_up --name_normal_dir 90_DEG --name_wandb_experiment patchcore_twoup_v1 --name two_up
```

### ReverseDistillation

```bash
python train_anomalib/train_reversedistillation_anomalib.py --dataset_root C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/dataset/images_lego_256/two_up --name_normal_dir 90_DEG --name_wandb_experiment revdist_twoup_v1 --name two_up --max_epochs 100 --patience 10
python train_anomalib/train_reversedistillation_anomalib.py --dataset_root C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/dataset/images_wood/a-no-black --name_normal_dir normal --name_wandb_experiment revdist_wood_a_no_black_v1 --name a-no-black --max_epochs 100 --patience 10
```

### EfficientAD

```bash
python train_anomalib/train_efficientAD_anomalib.py --dataset_root C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/dataset/images_wood/d-no-black --name_normal_dir normal --name_wandb_experiment effAD_d_no_black_v1 --name d-no-black --max_epochs 100 --patience 10 
```

### FastFlow

```bash
python train_anomalib/train_fastflow_anomalib.py --dataset_root C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/dataset/images_wood/d-no-black --name_normal_dir normal --name_wandb_experiment effAD_d_no_black_v1 --name d-no-black --max_epochs 100 --patience 10
```

## Test image models

### Patchcore

```bash
python infer_anomalib/test_model_patchcore_gpu.py --device cuda --path_torch_model C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/results/Patchcore/two_up/v1/weights/lightning/model.ckpt --path_dataset C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/dataset/images_lego_256/two_up --name two_up --dir_result C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/results/Patchcore/two_up/v1
```

### ReverseDistillation

```bash
python infer_anomalib/test_model_reversedistillation_gpu.py --device cuda --path_torch_model C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/results/ReverseDistillation/two_up/v0/weights/lightning/model.ckpt --path_dataset C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/dataset/images_lego_256/two_up --name two_up --dir_result C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/results/ReverseDistillation/two_up/v0
python infer_anomalib/test_model_reversedistillation_gpu.py --device cuda --normal_dir normal --path_torch_model C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/results/ReverseDistillation/a-no-black/v0/weights/lightning/model.ckpt --path_dataset C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/dataset/images_wood/a-no-black --name a-no-black --dir_result C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/results/ReverseDistillation/a-no-black/v0
```

### FastFlow

```bash
python infer_anomalib/test_model_fastflow_gpu.py --device cuda --normal_dir normal --path_torch_model C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/results/Fastflow/b-no-black/v0/weights/lightning/model.ckpt --path_dataset C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/dataset/images_wood/b-no-black --name b-no-black --dir_result C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/results/Fastflow/b-no-black/v0
```
