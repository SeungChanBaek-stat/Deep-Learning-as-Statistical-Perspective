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
import pandas as pd
import os



class GradCAMTest:
    def __init__(self, gradcam, idx_dict, img_list, bs, num_bs):
        self.gradcam = gradcam
        self.idx_dict = idx_dict
        self.result_list = []
        self.img_list = img_list
        self.bs = bs
        self.num_bs = num_bs
        
    def gradcam_heatmap_calc(self):
        img_list = self.img_list
        result_list = self.result_list
        for img_path_ in img_list:
            result_dict = {}
            img_tensor = self.gradcam.load_img(img_path=img_path_)
            if img_tensor.dtype != self.gradcam.dtype:
                img_tensor = img_tensor.to(self.gradcam.dtype)
            result_dict["img_path"] = img_path_
            result_dict["grad_cam_heatmap"] = []
            result_dict["out_class"] = []
            for batch_idx in range(self.num_bs):
                img_tensors = torch.cat([img_tensor for _ in range(self.bs)])
                #heatmap, predicted_class, img_np = self.model.run_grad_cam(img_tensor=img_tensors)
                # 이 부분 수정하기 
                heatmap, predicted_class, img_np = self.gradcam.run_grad_cam(img_path=img_path_, img_class=99)
                print("predicted_class : ", predicted_class)
                out_class = [self.idx_dict[pred_cls.item()] for pred_cls in predicted_class]
                print("out_class : ", out_class)
                grad_cam_heatmap = self.gradcam.show_cam_on_image(img=img_np, mask=heatmap, use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.5)
                result_dict["grad_cam_heatmap"].append(grad_cam_heatmap)
                result_dict["out_class"] += out_class
            result_dict["grad_cam_arr"] = np.stack(result_dict["grad_cam_heatmap"])
            result_list.append(result_dict)
            
        return result_list
    
    def gradcam_heatmap_visualize(self, result_list, save_plot_dir=None):
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
            save_filename = os.path.join(save_plot_dir, f"hyena_{idx}.png")
            plt.savefig(save_filename, bbox_inches='tight')
            plt.close(fig)        