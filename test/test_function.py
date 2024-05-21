import sys, os
# sys.path.append('/content/drive/MyDrive/Deep-Learning-as-Statistical-Perspective')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'prerequisites')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'gradcam')))
# sys.path.append(os.pardir)
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from prerequisites import ClassIdxToName
from prerequisites import torch_fix_seed
# from GradCAM_original import GradCAM
# from GradCAM_original_test import GradCAMTest
import pandas as pd

# VGG-16 사용을 위한 클래스 지정 (1000개)
idx_dict = ClassIdxToName()
search_results = idx_dict.search_word("fish")
print(search_results)


print(idx_dict[0]) # tench가 출력되어야 함

# 시드 고정
SEED = 77
torch_fix_seed(SEED)

weights = models.VGG16_Weights.IMAGENET1K_V1 # 이 코드에 사용하려는 모델 가중치 불러오기
model = models.vgg16(weights=weights) # 불러온 가중치 모델에 장착

print(model)


from tqdm import tqdm

# 이미지 4장에 대해서 GradCAM 테스트

device = torch.device("cuda")
dtype = torch.float32
gradcam = GradCAM(model=model, device=device, dtype=dtype)

img_path1 = '/content/drive/MyDrive/Deep-Learning-as-Statistical-Perspective/Images/test/gradcam_original_test_small/img_1.JPEG' # 하얀 늑대 사진
img_path2 = '/content/drive/MyDrive/Deep-Learning-as-Statistical-Perspective/Images/test/gradcam_original_test_small/img_2.JPEG' # 개 사진
img_path3 = '/content/drive/MyDrive/Deep-Learning-as-Statistical-Perspective/Images/test/gradcam_original_test_small/img_3.JPEG' # 개 2마리 사진
img_path4 = '/content/drive/MyDrive/Deep-Learning-as-Statistical-Perspective/Images/test/gradcam_original_test_small/img_4.png'
img_list = [img_path1, img_path2, img_path3, img_path4]

# 결과를 저장할 디렉토리 경로 설정
results_dir = "/content/drive/MyDrive/Deep-Learning-as-Statistical-Perspective/test/results"
# 디렉토리가 존재하지 않으면 생성
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
bs = 1
num_bs = 1
gradcam_test_small = GradCAMTest(gradcam=gradcam, idx_dict=idx_dict, img_list=img_list, bs=bs, num_bs=num_bs)
result_list = gradcam_test_small.gradcam_heatmap_calc()
# gradcam_test_small.gradcam_heatmap_visualize(result_list=result_list, save_plot_dir=results_dir)


