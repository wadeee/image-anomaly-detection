# Memo

## Installation

Build the environment based on python3.10
```bash
conda create -n image-anomaly-detection python=3.10
conda activate image-anomaly-detection
```

install all the packages using

```bash
pip install -r requirements-custom.txt
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

if CUDA is not present, then comment torch==2.1.2+cu121 and torchvision==0.16.2+cu121 and use torch==2.1.2 and torchvision==0.16.2. Then install Anomalib
```bash
# pip install anomalib==1.2.0
anomalib install
```

## Dataset

You can download the full dataset from the link

https://drive.google.com/file/d/1YgS897POEal8mkP3kngwcV5LI9vuyPuJ/view?usp=sharing

### Patchcore

```bash
python train_anomalib/train_patchcore_anomalib.py --dataset_root C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/dataset/images_lego_256/two_up --name_normal_dir 90_DEG --name_wandb_experiment patchcore_twoup_v1 --name two_up
python train_anomalib/train_patchcore_anomalib.py --dataset_root /C/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/dataset/images_lego_256/two_up --name_normal_dir 90_DEG --name_wandb_experiment patchcore_twoup_v1 --name two_up
```


## Test image models

Once a model has been train, you can test the model in order to find the confusion matrix and
how are distributed the scores for the normal and abnormal classification. You can use the following script

```bash
python infer_anomalib/test_model.py --path_torch_model C:/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/results/ReverseDistillation/one_up/v0/weights/torch/model.pt --path_dataset /home/enrico/Projects/Image_Anomaly_Detection/dataset/images_lego_256/one_up --name one_up --dir_result /home/enrico/Projects/Image_Anomaly_Detection/results/ReverseDistillation/one_up/v0
python infer_anomalib/test_model.py --path_torch_model /C/Users/Wadec/Documents/Projects/Image_Anomaly_Detection/results/ReverseDistillation/one_up/v0/weights/torch/model.pt --path_dataset /home/enrico/Projects/Image_Anomaly_Detection/dataset/images_lego_256/one_up --name one_up --dir_result /home/enrico/Projects/Image_Anomaly_Detection/results/ReverseDistillation/one_up/v0
```
