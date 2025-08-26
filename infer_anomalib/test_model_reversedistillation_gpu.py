import sys
from PIL import Image
from anomalib.data.utils import ValSplitMode
from anomalib.models import ReverseDistillation

sys.path.append('')

from anomalib.data.image.folder import Folder
from anomalib import TaskType

import numpy as np
from torch import as_tensor
from torchvision.transforms.v2.functional import to_dtype, to_image
import torch
import argparse
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm


class2label = {"NORMAL": 0, "ABNORMAL": 1}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_torch_model', type=str, help='Path to the torch model')
    parser.add_argument('--path_dataset', type=str, help='Path to the image to analyze')
    parser.add_argument('--name', type=str, help='Name the current dataset')
    parser.add_argument('--dir_result', type=str, help='Directory where to store the results')
    parser.add_argument('--image_size', type=int, default=256, help='Size of the image')
    parser.add_argument('--normal_dir', type=str, default="90_DEG", help='Name of the normal directory')
    # 新增GPU相关参数
    parser.add_argument('--device', type=str, default='auto', help='Device to use: auto, cpu, cuda, cuda:0, etc.')

    opt = parser.parse_args()

    # 设置设备
    if opt.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(opt.device)

    # load config file
    path_torch_model = opt.path_torch_model
    path_dataset = opt.path_dataset
    dir_result = opt.dir_result
    name = opt.name
    image_size = int(opt.image_size)
    normal_dir = opt.normal_dir

    # 直接从checkpoint加载模型
    model = ReverseDistillation.load_from_checkpoint(path_torch_model)
    model.eval()

    # 将模型移动到指定设备
    model = model.to(device)

    # load the datamodule
    datamodule = Folder(
        name=name,
        root=path_dataset,
        normal_dir=normal_dir,
        abnormal_dir="abnormal",
        task=TaskType.CLASSIFICATION,
        seed=42,
        val_split_mode=ValSplitMode.FROM_TEST,  # default value
        val_split_ratio=0.5,  # default value
        #image_size=(image_size,image_size)
    )

    # Setup the datamodule
    datamodule.setup()

    # take the test dataset
    test_dataset = datamodule.test_data.samples

    # create the dataset to store the results
    df_results = pd.DataFrame(columns=['IMG_PATH', 'TRUE_CATEGORY', 'TRUE_LABEL', 'PRED_CATEGORY', 'PRED_LABEL', 'PRED_SCORE', 'DEG'])

    # list to collect the true and pred labels
    true_label_list = []
    pred_label_list = []

    # iter over the test dataset
    for index, row in tqdm(test_dataset.iterrows(), total=test_dataset.shape[0], desc="test dataset"):
        image_path = row["image_path"]
        true_label = row["label_index"]

        deg = os.path.basename(os.path.dirname(image_path))

        image = Image.open(image_path).convert("RGB")
        image = image.resize((image_size, image_size))
        image = to_dtype(to_image(image), torch.float32, scale=True) if as_tensor else np.array(image) / 255.0

        # 确保图像数据格式正确并移动到指定设备
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:  # 如果是3D张量 (C, H, W)
                image_batch = image.unsqueeze(0).to(device)
            else:  # 如果已经是4D张量 (B, C, H, W)
                image_batch = image.to(device)
        else:
            # 如果是numpy数组，转换为torch张量
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
            image_batch = image.unsqueeze(0).to(device)

        # model.eval()
        batch = {"image": image_batch.float()}
        result = model.predict_step(batch, batch_idx=0)
        # result = model.predict_step(image_batch.float(), batch_idx=0)
        #
        true_category = ""
        if true_label == 0:
            true_category = "NORMAL"
        elif true_label == 1:
            true_category = "ABNORMAL"
        #
        # pred_category = ""
        # pred_score = 0
        #
        # # 获取预测分数
        # print('result')
        # # print(result)
        # print('result keys:', result.keys())
        # pred_score_raw = result['pred_scores'].cpu().item()
        # threshold = 49  # 根据你的模型调整

        # 对于ReverseDistillation，通常需要从anomaly_maps中提取分数
        if 'anomaly_maps' in result:
            # 获取异常图
            anomaly_map = result['anomaly_maps']

            # 方法1: 取异常图的最大值作为异常分数
            pred_score_raw = anomaly_map.max().cpu().item()

            # 方法2: 取异常图的平均值作为异常分数
            # pred_score_raw = anomaly_map.mean().cpu().item()

            # 方法3: 如果需要更复杂的处理，可以先调整异常图大小再取最大值
            # from torch.nn.functional import interpolate
            # resized_map = interpolate(anomaly_map.unsqueeze(0), size=(image_size, image_size), mode='bilinear')
            # pred_score_raw = resized_map.max().cpu().item()

        elif 'pred_scores' in result:
            # 如果确实有pred_scores（某些版本的实现可能不同）
            pred_score_raw = result['pred_scores'].cpu().item()
        else:
            # 如果都没有，可能需要检查模型输出
            print("无法找到预测分数，可用键:", result.keys())
            pred_score_raw = 0  # 默认值

        # 设置合适的阈值 - 这个需要根据您的模型和数据集调整
        # 通常可以通过在验证集上计算得到合适的阈值
        threshold = 0.5  # 示例阈值，需要根据实际情况调整

        # 生成预测标签
        pred_label = 1 if pred_score_raw >= threshold else 0

        # 根据分数和阈值确定标签和类别
        if pred_score_raw < threshold:
            pred_label = 0
            pred_score = pred_score_raw / threshold  # 归一化分数
            pred_category = "NORMAL"
        else:
            pred_label = 1
            pred_score = min(pred_score_raw / threshold, 1.0)  # 归一化并限制最大值为1
            pred_category = "ABNORMAL"

        true_label_list.append(true_label)
        pred_label_list.append(pred_label)

        # 创建新行的 DataFrame
        new_row = pd.DataFrame({
            'IMG_PATH': [image_path],
            'TRUE_CATEGORY': [true_category],
            'TRUE_LABEL': [true_label],
            'PRED_CATEGORY': [pred_category],
            'PRED_LABEL': [pred_label],
            'PRED_SCORE': [pred_score_raw],
            'DEG': [deg]
        })

        # 然后在循环中添加数据
        df_results = pd.concat([df_results, new_row], ignore_index=True)

    print('Accuracy: ', accuracy_score(true_label_list, pred_label_list))
    print(metrics.classification_report(true_label_list, pred_label_list, zero_division=0))

    ## Plot and save the confusion matrix
    cm = metrics.confusion_matrix(true_label_list, pred_label_list)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.set(font_scale=1.3)  # Adjust to fit
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=class2label.keys(),
           yticklabels=class2label.keys())
    # plt.yticks(fontsize=10, rotation=0)
    plt.yticks(fontsize=11, rotation=-30, ha='right', rotation_mode='anchor')
    # plt.xticks(fontsize=10, rotation=90)
    plt.xticks(fontsize=11, rotation=30, ha='right', rotation_mode='anchor')

    fig.savefig(os.path.join(dir_result, name + "_confusion_matrix.png"))


    ## Check distribution score over the right classification
    df_results_correct = df_results[df_results["TRUE_CATEGORY"] == df_results["PRED_CATEGORY"]]
    # boxplot
    plt.figure(figsize=(15, 15))
    sns.boxplot(data=df_results_correct, x="DEG", y="PRED_SCORE")
    plt.xticks(rotation=45)
    plt.title('DEG distribution classification', fontsize=12)
    plt.savefig(os.path.join(dir_result, "plot_right_classification.png"))


    ## Check deg distribution over the error classification
    df_results_error = df_results[df_results["TRUE_CATEGORY"] != df_results["PRED_CATEGORY"]]
    df2 = df_results_error.groupby(['DEG']).size().reset_index(name='COUNT')
    plt.figure(figsize=(15, 15))
    # create grouped bar chart
    barplot = sns.barplot(x='DEG', y='COUNT', data=df2, orient='v').set(title='Number of error per DEG')
    plt.savefig(os.path.join(dir_result, "plot_bad_classification.png"))

    # save the csv with results
    df_results.to_csv(os.path.join(dir_result, name + "_result.csv"), index=False)