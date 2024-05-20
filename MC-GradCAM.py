import sys, os
sys.path.append('/content/drive/MyDrive/Deep-Learning-as-Statistical-Perspective')
sys.path.append(os.pardir)
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from class_idx_to_name import class_idx_to_name

# VGG-16 사용을 위한 클래스 지정 (1000개)
idx_dict = class_idx_to_name()

print(idx_dict()[0]) # tench가 출력되어야 함

weights = models.VGG16_Weights.IMAGENET1K_V1 # 이 코드에 사용하려는 모델 가중치 불러오기
model = models.vgg16(weights=weights) # 불러온 가중치 모델에 장착

print(model)