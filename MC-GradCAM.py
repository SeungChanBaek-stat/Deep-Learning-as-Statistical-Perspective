import sys, os
# sys.path.append('/content/drive/MyDrive/Deep-Learning-as-Statistical-Perspective')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'prerequisites')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'GradCAM')))
# sys.path.append(os.pardir)
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from prerequisites import class_idx_to_name
from prerequisites import torch_fix_seed
from GradCAM_original import GradCAM, GradCAM_TEST
import pandas as pd

# VGG-16 사용을 위한 클래스 지정 (1000개)
idx_dict = class_idx_to_name()

print(idx_dict[0]) # tench가 출력되어야 함

# 시드 고정
SEED = 77
torch_fix_seed(SEED)

weights = models.VGG16_Weights.IMAGENET1K_V1 # 이 코드에 사용하려는 모델 가중치 불러오기
model = models.vgg16(weights=weights) # 불러온 가중치 모델에 장착

print(model)


from tqdm import tqdm
import cv2
import os

device = torch.device("cuda")
dtype = torch.float32
gradcam = GradCAM(model=model, device=device, dtype=dtype)

# 이미지 4장에 대해서 GradCAM 테스트
gradcam_test_instance = GradCAM_TEST(gradcam=gradcam, idx_dict=idx_dict)
result_list = gradcam_test_instance.gradcam_test()


import os

# 결과를 저장할 디렉토리 경로 설정
results_dir = "/content/drive/MyDrive/Deep-Learning-as-Statistical-Perspective/Images/gradcam_original_test"

# 디렉토리가 존재하지 않으면 생성
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

for idx in range(len(result_list)):
    img_path = result_list[idx]["img_path"]
    out_class = result_list[idx]["out_class"]
    grad_cam_arr = result_list[idx]["grad_cam_arr"]

    img_original = Image.open(img_path).convert('RGB')
    out_series = pd.Series(out_class).value_counts()
    out_df = pd.DataFrame(out_series, columns=["count"], index=out_series.index).reset_index()
    predicted_label = pd.Series(out_class).value_counts().index[0]

    grad_cam_heatmap = np.squeeze(grad_cam_arr, axis=0)
    fig, axes = plt.subplots(1, 2, figsize = (8, 8))
    axes = axes.flatten()
    axes[0].imshow(img_original)
    axes[0].set_title(f"Predicted as {predicted_label}")
    #axes[1].imshow(np.linalg.norm(grad_cam_heatmap, axis=-1, ord=2))
    axes[1].imshow(grad_cam_heatmap)
    axes[1].set_title(f"GradCAM")
    for i in range(2):
        axes[i].axis("off")
    fig.tight_layout()

    output_path = os.path.join(results_dir, f"result_{idx + 1}.png")
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")

# # 저장 경로 설정
# path_to_save_grad = '/content/drive/MyDrive/DLSP_Uncertainty_Quantification/MC gradcam/20231205 v3/gradcamvalue'
# path_to_save_heatmap = '/content/drive/MyDrive/DLSP_Uncertainty_Quantification/MC gradcam/20231205 v3/heatmap'

# # Ensure save directories exist
# os.makedirs(path_to_save_grad, exist_ok=True)
# os.makedirs(path_to_save_heatmap, exist_ok=True)