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
from GradCAM_original import GradCAM

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
gradcam = GradCAM(model, device=device, dtype=dtype)

# 이미지 경로 설정
img_path1='/content/drive/MyDrive/DLSP_Uncertainty_Quantification/ImageNet_val_image/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00000027.JPEG' # 하얀 늑대 사진
img_path2='/content/drive/MyDrive/DLSP_Uncertainty_Quantification/ImageNet_val_image/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00001624.JPEG' # 개 사진
img_path3='/content/drive/MyDrive/DLSP_Uncertainty_Quantification/ImageNet_val_image/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00000228.JPEG' # 개 2마리 사진
img_path4='/content/drive/MyDrive/DLSP_Uncertainty_Quantification/Final/20231227/pytorch-grad-cam-test/both.png'


imgtest1 = Image.open(img_path1)
imgtest1_np = np.array(imgtest1)
print(np.shape(imgtest1_np))


# # 이미지 크기 확인
# #width, height = imgtest1.size
# #print("Width:", width, "Height:", height)

# # 저장 경로 설정
# path_to_save_grad = '/content/drive/MyDrive/DLSP_Uncertainty_Quantification/MC gradcam/20231205 v3/gradcamvalue'
# path_to_save_heatmap = '/content/drive/MyDrive/DLSP_Uncertainty_Quantification/MC gradcam/20231205 v3/heatmap'

# # Ensure save directories exist
# os.makedirs(path_to_save_grad, exist_ok=True)
# os.makedirs(path_to_save_heatmap, exist_ok=True)