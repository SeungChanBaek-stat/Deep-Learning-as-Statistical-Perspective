import sys, os
# sys.path.append('/content/drive/MyDrive/Deep-Learning-as-Statistical-Perspective')
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'prerequisites')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'gradcam')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.pardir)
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from prerequisites.prerequisites import ClassIdxToName, torch_fix_seed
from MCGradCAM.MCGradCAM import MCGradCAM
from MCGradCAM.MCGradCAM_visual import MCGradCAMVisualize

import pandas as pd

# VGG-16 사용을 위한 클래스 지정 (1000개)
idx_dict = ClassIdxToName()
search_results = idx_dict.search_word("fish")
print(search_results)

print(type(idx_dict))
print(idx_dict.dict_modified[0])# tench가 출력되어야 함

# 시드 고정
SEED = 77
torch_fix_seed(SEED)

# vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
weights = models.VGG16_Weights.IMAGENET1K_V1
vgg16 = models.vgg16(weights=weights)
vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

from tqdm import tqdm

# 이미지 4장에 대해서 GradCAM 테스트

device = torch.device("cuda")
dtype = torch.float32
gradcam = MCGradCAM(model=vgg16, device=device, dtype=dtype)


# 이미지 경로 설정
small_test_dir = "c:\\Users\\AAA\\Deep-Learning-as-Statistical-Perspective\\data\\gradcam_original_test_small\\test"
img_path1 = small_test_dir + '\\img_1.JPEG' # 하얀 늑대 사진
img_path2 = small_test_dir + '\\img_2.JPEG' # 개 사진
img_path3 = small_test_dir + '\\img_3.JPEG' # 개 2마리 사진
img_path4 = small_test_dir + '\\img_4.png'
img_list = [img_path1, img_path2, img_path3, img_path4]


# # 결과를 저장할 디렉토리 경로 설정
results_dir = "/content/drive/MyDrive/Deep-Learning-as-Statistical-Perspective/test/results"
# # 디렉토리가 존재하지 않으면 생성
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)
T = 2500
# T = bs * num_bs
num_bs = 5
bs = int(T / num_bs) # 500
mcgradcam = MCGradCAM(model=vgg19, device=device, dtype=dtype)
mcgradcam_test_small = MCGradCAMVisualize(mcgradcam=mcgradcam, idx_dict=idx_dict, img_list=img_list, bs=bs, T=T)
result_list = mcgradcam_test_small.mcgradcam_heatmap_calc()
# mcgradcam_test_small.gradcam_heatmap_visualize(result_list=result_list, save_plot_dir=results_dir)
# mcgradcam.run_mc_grad_cam(img_path=img_path1, img_tensor=None, nmc=T, bs=bs)