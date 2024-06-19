import torch
from torchvision import models, transforms
from PIL import Image
from torch import nn,Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np

class MCGradCAM:
    def __init__(self, model, dtype=torch.float32, device=torch.device("cpu")):
        self.dtype = dtype
        self.device = device

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
        self.classifier.train()
        self.to(device, dtype=dtype)
        self.classifier_train = True

    def to(self, device, dtype=torch.float32):
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


    # 클래스 스코어 추출 함수
    def get_class_score(self, img_tensor, bs=1):
        # consider img_tensor = [1, n_c , imgd1, imgd2]
        with torch.cuda.amp.autocast():
            self.feature_extractor.zero_grad()
            self.classifier.zero_grad()

            # 이미지 전처리 및 requires_grad 설정
            img_tensor.requires_grad_(True)
            with torch.no_grad():
                Ak = self.feature_extractor(img_tensor)
            # Ak: [1, nc, fs, fs]
            Aks = Ak.clone().repeat([bs, 1, 1, 1])
            Aks.requires_grad_(True)
            out = self.classifier(Aks)
            _, predicted_class = out.max(1)
            score = out[:, predicted_class]
            score_out = score.sum()
            score_out.backward()
            grad_Aks = Aks.grad
            # print(f"Aks shape = {Aks.shape}")
            # print(f"grad_Aks shape = {grad_Aks.shape}")

        return predicted_class, score, Aks, grad_Aks, out

    # Grad-CAM 계산기
    def calculate_grad_cam(self, Ak, gradients, target_size):
        # 그라디언트의 글로벌 평균 계산 (by channel)
        alpha_c_k = torch.mean(gradients, dim=[2, 3])

        heatmap = torch.einsum('ij,ijkl->ikl', alpha_c_k, Ak)

        # ReLU 적용 - 음수 값 제거
        heatmap = torch.relu(heatmap)


        # heatmap을 [0, 1] 범위로 배치 단위로 정규화
        heatmap_min = heatmap.view(heatmap.size(0), -1).min(dim=1, keepdim=True)[0]
        heatmap_max = heatmap.view(heatmap.size(0), -1).max(dim=1, keepdim=True)[0]
        heatmap = (heatmap - heatmap_min.view(-1, 1, 1)) / (heatmap_max.view(-1, 1, 1) - heatmap_min.view(-1, 1, 1))

        # # heatmap을 [0, 1] 범위로 정규화
        # heatmap_min = torch.min(heatmap)
        # heatmap_max = torch.max(heatmap)
        # heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)        
        # print(f"after normalizing heatmap shape = {heatmap.shape}")

        # heatmap을 target_size로 리사이징
        # print("resize 이전 히트맵 크기 (bs, width, height): ", np.shape(heatmap))
        heatmap = heatmap.unsqueeze(1)
        # print("unsqueeze 이후 히트맵 크기 (채널 차원 추가 : bs, c, width, height) : ", np.shape(heatmap))
        heatmap = F.interpolate(heatmap, size=target_size, mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze(1)
        # print("resize 이후 히트맵 크기 (채널 차원 제거 : bs, width, height) : ", np.shape(heatmap))

        #return heatmap.cpu().detach().numpy().astype(np.float32)
        return heatmap



    def run_mc_grad_cam(self, img_path=None, img_tensor=None, nmc=10, bs=None):
        img = Image.open(img_path)

        if img_path is not None:
            img_tensor = self.load_img(img_path)
            img_np = np.array(img)
        else :
            img_np = np.array(img)

        if bs is None:
            bs = nmc

        if not self.classifier_train:
            self.to_train_mode()

        if nmc % bs != 0:
            raise NotImplementedError
        else:
            num_batch = nmc // bs

        # assume img_tensor = [img_tensor_1, img_tensor_2, ...] = [num_img, n_channel, img_dim1, img_dim2]
        # 클래스와 해당 스코어 추출

        target_size = np.shape(img_np)
        # print("targrt_size 변형전 : ", target_size)
        if len(target_size) == 3:
            target_size = np.shape(img_np[:,:,0])
        else :
            target_size = np.shape(img_np)


        target_size = np.shape(img_np[:,:,0])
        # print("target_size 변형후 : ", target_size)

        pred_list = []
        grad_cam_list = []
        grad_cam_mean = None
        grad_cam_std = None

        for mini_batch in range(num_batch):
            predicted_class, score, Aks, grad_Aks, out = self.get_class_score(img_tensor, bs)
            grad_cam_list.append(self.calculate_grad_cam(Aks, grad_Aks, target_size))
            pred_list.append(predicted_class)

        grad_cam_t = torch.cat(grad_cam_list)
        pred_t = torch.cat(pred_list)



        return grad_cam_t, pred_t, img_np

    def to_eval_mode(self):
        if self.classifier_train:
            self.classifier.eval()
            self.classifier_train = False

    def to_train_mode(self):
        self.classifier.train()
        self.classifier_train = True


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
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().detach().numpy().astype(np.float32)
        else:
            mask = mask.astype(np.float32)

        mask = (mask - mask.min()) / (mask.max() - mask.min())
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



    def show_uncertainty_on_image(self, img: np.ndarray, mask: np.ndarray, use_rgb: bool = False, colormap: int = cv2.COLORMAP_JET, image_weight: float = 0.5) -> np.ndarray:
        """ This function overlays the cam mask on the image as an heatmap.
        By default the heatmap is in BGR format.

        :param img: The base image in RGB or BGR format.
        :param mask: The cam mask.
        :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
        :param colormap: The OpenCV colormap to be used.
        :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
        :returns: The default image with the cam overlay.
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().detach().numpy().astype(np.float32)
        else:
            mask = mask.astype(np.float32)

        mask = (mask - mask.min()) / (mask.max() - mask.min())
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