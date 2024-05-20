import torch
from torchvision import models, transforms
from PIL import Image
from torch import nn,Tensor
import torch.nn.functional as F
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import cv2
import numpy as np
from torchvision.transforms import Compose, Normalize, ToTensor
from typing import List, Dict
import math

class GradCAM:
    def __init__(self, model, dtype=torch.float32, device=torch.device("cpu")):
        self.feature_maps = None
        self.dtype = dtype
        self.device = device

        # devide model components
        self.feature_extractor = nn.Sequential(
            model.features,
            model.avgpool)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            model.classifier
        )
        self.feature_extractor.eval()
        self.classifier.eval()
        self.to(device, dtype=dtype)

    def to(self, device, dtype=torch.float32):
        self.feature_extractor = self.feature_extractor.to(device, dtype=dtype)
        self.classifier = self.classifier.to(device, dtype=dtype)


    # 클래스 스코어 추출 함수
    def get_class_score(self, img_tensor):

        with torch.cuda.amp.autocast():
          self.feature_extractor.zero_grad()
          self.classifier.zero_grad()

          # 이미지 전처리 및 requires_grad 설정
          img_tensor.requires_grad_(True)
          with torch.no_grad():
            Ak = self.feature_extractor(img_tensor)
          Ak.requires_grad_(True)
          out = self.classifier(Ak)
          _, predicted_class = out.max(1)
          score = out[:, predicted_class]
          score_out = score.sum()
          score_out.backward()
          grad_Ak = Ak.grad

        return predicted_class, score, Ak, grad_Ak, out


    # 클래스 스코어 추출 함수
    def get_class_score_ranking(self, img_tensor, ranking):

        with torch.cuda.amp.autocast():
          self.feature_extractor.zero_grad()
          self.classifier.zero_grad()

          # 이미지 전처리 및 requires_grad 설정
          img_tensor.requires_grad_(True)
          with torch.no_grad():
            Ak = self.feature_extractor(img_tensor)
          Ak.requires_grad_(True)
          out = self.classifier(Ak)
          _, predicted_class = out.max(ranking)
          score = out[:, predicted_class]
          score_out = score.sum()
          score_out.backward()
          grad_Ak = Ak.grad

        return predicted_class, score, Ak, grad_Ak, out

    # 클래스 스코어 추출 함수
    def get_class_score_img_given(self, img_tensor, img_class):

        with torch.cuda.amp.autocast():
          self.feature_extractor.zero_grad()
          self.classifier.zero_grad()

          # 이미지 전처리 및 requires_grad 설정
          img_tensor.requires_grad_(True)
          with torch.no_grad():
            Ak = self.feature_extractor(img_tensor)
          Ak.requires_grad_(True)
          out = self.classifier(Ak)
          predicted_class = img_class
          score = out[:, predicted_class]
          score_out = score.sum()
          score_out.backward()
          grad_Ak = Ak.grad

        return predicted_class, score, Ak, grad_Ak, out


    # Grad-CAM 계산기
    def calculate_grad_cam(self, Ak, gradients, target_size):
        # 그라디언트의 글로벌 평균 계산 (by channel)
        alpha_c_k = torch.mean(gradients, dim=[2, 3])

        # # feature maps에 그라디언트 가중치를 곱하여 클래스의 activation map 생성
        # weighted_feature_maps = Ak * alpha_c_k[:, None, None]
        # # [bs, n_channel, feat_dim, feat_dim]

        # # 클래스별 가중 feature maps의 채널별 합산
        heatmap = torch.einsum('ij,ijkl->ikl', alpha_c_k, Ak)

        # ReLU 적용 - 음수 값 제거
        heatmap = torch.relu(heatmap)

        # heatmap을 [0, 1] 범위로 정규화
        heatmap_min = torch.min(heatmap)
        heatmap_max = torch.max(heatmap)
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

        # heatmap을 target_size로 리사이징
        # print("resize 이전 히트맵 크기 : ", np.shape(heatmap))
        heatmap = heatmap.squeeze(0)
        # print("squeeze 이후 히트맵 크기 : ", np.shape(heatmap))
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        # print("unsqueeze 이후 히트맵 크기 : ", np.shape(heatmap))
        heatmap = F.interpolate(heatmap, size=target_size, mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze(0).squeeze(0)
        # print("resize 이후 히트맵 크기 : ", np.shape(heatmap))

        #return heatmap.cpu().detach().numpy().astype(np.float32)
        return heatmap

    def load_img(self, img_path):
        # 이미지 전처리 및 requires_grad 설정
        img = Image.open(img_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_preprocessed = preprocess(img).to(self.dtype)  # 여기에 .to(self.dtype) 추가
        img_preprocessed = img_preprocessed.unsqueeze(0).to(self.device, self.dtype)  # 이미지를 모델의 데이터 타입과 디바이스로 이동
        img_preprocessed.requires_grad_(True)
        return img_preprocessed


    def run_grad_cam(self, img_path=None, img_tensor=None, img_class=None, ranking=None):
        img = Image.open(img_path)
        if img_path is not None:
            img_tensor = self.load_img(img_path)
            img_np = np.array(img)
        else :
            img_np = np.array(img)

        if img_class is None:
            # print("클래스 is Not Given!")
            predicted_class, score, Ak, grad_Ak, out = self.get_class_score_ranking(img_tensor, ranking)
            # print("predicted class 타입 : ", type(predicted_class))
            # print("predicted class 사이즈 : ", predicted_class.shape )

            target_size = np.shape(img_np)
            # print("targrt_size 변형전 : ", target_size)
            if len(target_size) == 3:
                target_size = np.shape(img_np[:,:,0])
            else :
                target_size = np.shape(img_np)

            # target_size = np.shape(img_np[:,:,0])

            # print("target_size 변형후 : ", target_size)


        else :
            # print("클래스 is Given!")
            predicted_class = torch.tensor([img_class], device=self.device)
            predicted_class, score, Ak, grad_Ak, out = self.get_class_score_img_given(img_tensor, predicted_class)
            # print("predicted class 타입 : ", type(predicted_class))
            # print("predicted class 사이즈 : ", predicted_class.shape )

            target_size = np.shape(img_np)
            # print("targrt_size 변형전 : ", target_size)
            if len(target_size) == 3:
                target_size = np.shape(img_np[:,:,0])
            else :
                target_size = np.shape(img_np)

            # target_size = np.shape(img_np[:,:,0])

            # print("target_size 변형후 : ", target_size)



        return self.calculate_grad_cam(Ak, grad_Ak, target_size), predicted_class, img_np




    def show_cam_on_image(self, img: np.ndarray, mask: np.ndarray, use_rgb: bool = False, colormap: int = cv2.COLORMAP_JET, image_weight: float = 0.5) -> np.ndarray:
        """ This function overlays the cam mask on the image as an heatmap.
        By default the heatmap is in BGR format.

        :param img: The base image in RGB or BGR format.
        :param mask: The cam mask.
        :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
        :param colormap: The OpenCV colormap to be used.
        :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
        :returns: The default image with the cam overlay.
        """
        mask = mask.cpu().detach().numpy().astype(np.float32)
        img = img / 255.0

        heatmap = cv2.applyColorMap(np.uint8(255.0 * mask), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255.0

        if np.max(img) > 1:
            raise Exception(
                "The input image should np.float32 in the range [0, 1]")

        if image_weight < 0 or image_weight > 1:
            raise Exception(
                f"image_weight should be in the range [0, 1].\
                    Got: {image_weight}")

        # 원본 이미지가 흑백인 경우 채널 차원을 추가
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]  # (H, W) -> (H, W, 1)

        # 이미지가 흑백이면서 채널 차원을 추가한 경우, 이 차원을 히트맵의 채널 수에 맞게 복제
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)  # (H, W, 1) -> (H, W, 3)

        cam = (1 - image_weight) * heatmap + image_weight * img
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)
    
    
    

