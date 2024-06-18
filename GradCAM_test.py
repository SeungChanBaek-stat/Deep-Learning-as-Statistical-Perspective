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
from GradCAM.GradCAM import GradCAM
from GradCAM.GradCAM_visual import GradCAMVisualize

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
gradcam = GradCAM(model=vgg16, device=device, dtype=dtype)


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
bs = 1
num_bs = 1
gradcam = GradCAM(model=vgg19, device=device, dtype=dtype, mode="top-1")
gradcam_test_small = GradCAMVisualize(gradcam=gradcam, idx_dict=idx_dict, img_list=img_list, bs=bs, num_bs=num_bs)
result_list = gradcam_test_small.gradcam_heatmap_calc()
gradcam_test_small.gradcam_heatmap_visualize(result_list=result_list, save_plot_dir=results_dir)






# # 결과를 저장할 디렉토리 경로 설정
results_dir = "/content/drive/MyDrive/Deep-Learning-as-Statistical-Perspective/test/results"
# # 디렉토리가 존재하지 않으면 생성
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)
bs = 1
num_bs = 1
gradcam = GradCAM(model=vgg19, device=device, dtype=dtype, mode="top-3")
gradcam_test_small = GradCAMVisualize(gradcam=gradcam, idx_dict=idx_dict, img_list=img_list, bs=bs, num_bs=num_bs)
result_list = gradcam_test_small.gradcam_heatmap_calc()
gradcam_test_small.gradcam_heatmap_visualize(result_list=result_list, save_plot_dir=results_dir)







# # 원본 이미지 202장에 대해서 GradCAM 테스트

# device = torch.device("cuda")
# dtype = torch.bfloat16
# gradcam = GradCAM(model=model, device=device, dtype=dtype)

# # 결과를 저장할 디렉토리 경로 설정
# results_dir = "/content/drive/MyDrive/Deep-Learning-as-Statistical-Perspective/Images/gradcam_original_test_large/test_or_results"

# # 디렉토리가 존재하지 않으면 생성
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)

# path1= results_dir + "/test_or"
# lst1 = os.listdir(path1)
# lst1 = sorted(lst1)
# img_list_O = []
# for i, filename in enumerate(lst1, start=1):
#     img_path = f"{path1}/{filename}"
#     globals()[f'img_path{i}'] = img_path
#     img_list_O.append(img_path)

# bs = 1
# num_bs = 1
# gradcam_test_small = GradCAMVisualize(gradcam=gradcam, idx_dict=idx_dict, img_list=img_list_O, bs=bs, num_bs=num_bs)
# result_list = gradcam_test_small.gradcam_heatmap_calc()
# gradcam_test_small.gradcam_heatmap_visualize(result_list=result_list, save_plot_dir=results_dir)
    
    
    
    
    
   
    
# # 변형된 이미지 202장에 대해서 GradCAM 테스트

# device = torch.device("cuda")
# dtype = torch.bfloat16
# gradcam = GradCAM(model=model, device=device, dtype=dtype)

# # 결과를 저장할 디렉토리 경로 설정
# results_dir = "/content/drive/MyDrive/Deep-Learning-as-Statistical-Perspective/Images/gradcam_original_test_large/test_tf_results"

# # 디렉토리가 존재하지 않으면 생성
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)    


# path2= results_dir + "/test_tf"
# lst2 = os.listdir(path2)
# lst2 = sorted(lst2)
# img_list_T = []
# for i, filename in enumerate(lst2, start=1):
#     img_path = f"{path2}/{filename}"
#     globals()[f'img_path{i}'] = img_path
#     img_list_T.append(img_path)
    
# bs = 1
# num_bs = 1
# gradcam_test_small = GradCAMVisualize(gradcam=gradcam, idx_dict=idx_dict, img_list=img_list_T, bs=bs, num_bs=num_bs)
# result_list = gradcam_test_small.gradcam_heatmap_calc()
# gradcam_test_small.gradcam_heatmap_visualize(result_list=result_list, save_plot_dir=results_dir)
