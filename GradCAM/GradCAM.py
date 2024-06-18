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
    """GradCAM을 사용하여 모델의 주목 영역을 시각화하는 클래스.

    Attributes:
        model: CNN기반 모형.
        dtype: 데이터 타입 (기본값: torch.float32).
        device: 연산에 사용될 장치 (기본값: CPU).
        feature_maps: 모델의 활성화 맵.
        feature_extractor: 모델의 마지막 합성곱층의 활성화맵 추출 부분.
        classifier: 모델의 분류 부분.
        mode: "default" - 클래스를 정하고 gradcam 계산, "top-3" - 모델이 예측한 top3순위에 해당하는 gradcam 계산
    """
    def __init__(self, model, dtype=torch.float32, device=torch.device("cpu"), mode="default"):
        self.feature_maps = None
        self.dtype = dtype
        self.device = device
        self.mode = mode

        alexnet_family = ["alexnet"]
        vgg_family = ["vgg"]        
        resnet_family = ["resnet"]

        model_name = model.__class__.__name__.lower()

        if model_name == "alexnet":
            self.feature_extractor = nn.Sequential()
            feature_module_num = len(model.features)
            for i in range(feature_module_num - 2):
                self.feature_extractor.add_module(f'layer{i}', model.features[i])
            self.feature_extractor.add_module(f'layer{feature_module_num - 2}', nn.ReLU(inplace=False))
            self.classifier = nn.Sequential(
                model.features[12],
                model.avgpool,
                nn.Flatten(),
                model.classifier
            )
            print(f"alexnet family : {model_name}")
        elif model_name == "vgg":
            self.feature_extractor = nn.Sequential()
            feature_module_num = len(model.features)
            for i in range(feature_module_num - 2):
                self.feature_extractor.add_module(f'layer{i}', model.features[i])
            self.feature_extractor.add_module(f'layer{feature_module_num - 2}', nn.ReLU(inplace=False))
            self.classifier = nn.Sequential(
                model.features[feature_module_num - 1],
                model.avgpool,
                nn.Flatten(),
                model.classifier
            )
            print(f"vgg family : {model_name}")
        elif model_name == "resnet":
            self.feature_extractor = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4
            )
            self.classifier = nn.Sequential(
                model.avgpool,
                nn.Flatten(),
                model.fc
            )
            print(f"resnet family : {model_name}")
        else:
            print(f"Unknown model family: {model_name}")

        # print(self.feature_extractor)
        # print(self.classifier)

        self.feature_extractor.eval()
        self.classifier.eval()
        self.to(device, dtype=dtype)


    def to(self, device, dtype=torch.float32):
        """모델을 지정한 장치와 데이터 타입으로 이동"""
        self.feature_extractor = self.feature_extractor.to(device, dtype=dtype)
        self.classifier = self.classifier.to(device, dtype=dtype)


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


    def get_class_score_default(self, img_tensor, img_class):
        self.feature_extractor.zero_grad()
        self.classifier.zero_grad()

        # 이미지 전처리 및 requires_grad 설정
        img_tensor.requires_grad_(True)
        with torch.no_grad():
            Ak = self.feature_extractor(img_tensor)

        Ak.requires_grad_(True)

        # 일시적으로 첫 번째 ReLU를 inplace=False로 변경
        if isinstance(self.classifier[0], nn.ReLU) and self.classifier[0].inplace:
            self.classifier[0].inplace = False

        try:
            with torch.cuda.amp.autocast():
                out = self.classifier(Ak)
                print(f"out shape after classifier: {out.shape}")
        except Exception as e:
            print(f"Error during classifier forward pass: {e}")
            return None
        finally:
            # 원래 상태로 복구
            if isinstance(self.classifier[0], nn.ReLU):
                self.classifier[0].inplace = True

        predicted_class = img_class
        score = out[:, predicted_class]
        score_out = score.sum()
        score_out.backward()
        grad_Ak = Ak.grad

        return predicted_class, score, Ak, grad_Ak, out

    def get_class_score_top_1(self, img_tensor):
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
        # print(f"grad_Ak_top-1 = {grad_Ak}")

        return predicted_class, score, Ak, grad_Ak, out



    def get_class_score_top_3(self, img_tensor):
        self.feature_extractor.zero_grad()
        self.classifier.zero_grad()

        # 이미지 전처리 및 requires_grad 설정
        img_tensor.requires_grad_(True)
        with torch.no_grad():
            Ak = self.feature_extractor(img_tensor)

        Ak.requires_grad_(True)

        # 일시적으로 첫 번째 ReLU를 inplace=False로 변경
        if isinstance(self.classifier[0], nn.ReLU) and self.classifier[0].inplace:
            self.classifier[0].inplace = False

        try:
            with torch.cuda.amp.autocast():
                out = self.classifier(Ak)
                print(f"out shape after classifier: {out.shape}")
        except Exception as e:
            print(f"Error during classifier forward pass: {e}")
            return None
        finally:
            # 원래 상태로 복구
            if isinstance(self.classifier[0], nn.ReLU):
                self.classifier[0].inplace = True

        # 상위 3개의 클래스를 찾기
        topk_scores, topk_indices = torch.topk(out, 3, dim=1)


        predicted_class_list = []
        score_list = []
        grad_Ak_list = []

        print(f"topk_indices = {topk_indices}")

        for indice in topk_indices.squeeze():
            predicted_class = indice
            score = out[:, predicted_class]
            score_out = score.sum()
            score_out.backward(retain_graph=True)
            grad_Ak = Ak.grad.clone()
            predicted_class_list.append(predicted_class.unsqueeze(0))
            score_list.append(score.unsqueeze(0))
            grad_Ak_list.append(grad_Ak.unsqueeze(0))

        predicted_class_tensor = torch.cat(predicted_class_list)
        score_tensor = torch.cat(score_list)
        grad_Ak_tensor = torch.cat(grad_Ak_list)
        # print(f"grad_Ak_top-1 = {grad_Ak_tensor[0]}")

        return predicted_class_tensor, score_tensor, Ak, grad_Ak_tensor, out

    # Grad-CAM 계산기
    def calculate_grad_cam(self, Ak, gradients, target_size):
        """
        feature maps에 그라디언트 가중치를 곱하여 클래스의 activation map 생성
        weighted_feature_maps = Ak * alpha_c_k[:, None, None]
        [bs, n_channel, feat_dim, feat_dim]
        """
        if self.mode == "top-3":
            alpha_c_k = torch.mean(gradients, dim = [3,4])
            heatmaps = torch.einsum('bij,ijkl->bikl', alpha_c_k, Ak)
            heatmaps = torch.relu(heatmaps)

            # heatmap을 [0, 1] 범위로 배치 단위로 정규화
            heatmap_min = heatmaps.view(heatmaps.size(0), -1).min(dim=1, keepdim=True)[0]
            heatmap_max = heatmaps.view(heatmaps.size(0), -1).max(dim=1, keepdim=True)[0]
            heatmaps = (heatmaps - heatmap_min[:, :, None, None]) / (heatmap_max[:, :, None, None] - heatmap_min[:, :, None, None])


            heatmaps = F.interpolate(heatmaps, size=target_size, mode='bilinear', align_corners=False)
            heatmaps = heatmaps.squeeze(1)

            return heatmaps


        else:
            # 그라디언트의 글로벌 평균 계산 (by channel)
            alpha_c_k = torch.mean(gradients, dim=[2, 3])

            # # 클래스별 가중 feature maps의 채널별 합산
            heatmap = torch.einsum('ij,ijkl->ikl', alpha_c_k, Ak)

            # ReLU 적용 - 음수 값 제거
            heatmap = torch.relu(heatmap)

            # heatmap을 [0, 1] 범위로 정규화
            heatmap_min = torch.min(heatmap)
            heatmap_max = torch.max(heatmap)
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

            # heatmap을 target_size로 리사이징
            heatmap = heatmap.squeeze(0)
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)
            heatmap = F.interpolate(heatmap, size=target_size, mode='bilinear', align_corners=False)
            heatmap = heatmap.squeeze(0).squeeze(0)

            return heatmap


    def run_grad_cam(self, img_tensor=None, img_np=None, img_class=None):
        target_size = np.shape(img_np)
        if len(target_size) == 3:
            target_size = np.shape(img_np[:,:,0])
        else :
            target_size = np.shape(img_np)

        # print("클래스 is Given!")
        if self.mode == "default":
            predicted_class = torch.tensor([img_class], device=self.device)
            predicted_class, score, Ak, grad_Ak, out = self.get_class_score_default(img_tensor, predicted_class)
            grad_cam = self.calculate_grad_cam(Ak, grad_Ak, target_size)
            #print(f"grad_cam_default shape = {grad_cam.shape}")
            return grad_cam, predicted_class

        elif self.mode == "top-1": # 여기수정하기
            predicted_class, score, Ak, grad_Ak, out = self.get_class_score_top_1(img_tensor)
            grad_cam_top_1 = self.calculate_grad_cam(Ak, grad_Ak, target_size)
            # print(f"grad_cam_top-1 = {grad_cam_top_1}")
            return grad_cam_top_1, predicted_class

        elif self.mode == "top-3":
            grad_cam_list = []
            pred_list = []

            predicted_class_tensor, score_tensor, Ak, grad_Ak_tensor, out = self.get_class_score_top_3(img_tensor)
            grad_cam_top_3 = self.calculate_grad_cam(Ak, grad_Ak_tensor, target_size)
            # print(f"grad_cam_top_1 = {grad_cam_top_3[0]}")
            return grad_cam_top_3, predicted_class_tensor




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