from PIL import Image
import numpy as np
import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os



class GradCAMVisualize:
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
            img = Image.open(img_path_)
            img_np = np.array(img)
            result_dict = {}
            img_tensor = self.gradcam.load_img(img_path=img_path_)
            if img_tensor.dtype != self.gradcam.dtype:
                img_tensor = img_tensor.to(self.gradcam.dtype)
            result_dict["img_path"] = img_path_
            result_dict["grad_cam_heatmap"] = []
            result_dict["grad_cam_heatmap_top-3"] = []
            result_dict["out_class"] = []
            result_dict["out_class_top-3"] = []
            for batch_idx in range(self.num_bs):
                img_tensors = torch.cat([img_tensor for _ in range(self.bs)])
                if self.gradcam.mode == "top-3":
                    heatmap_tensor, predicted_class_tensor = self.gradcam.run_grad_cam(img_tensor=img_tensors, img_np=img_np, img_class=270)
                    print("predicted_class : ", predicted_class_tensor, predicted_class_tensor.shape)
                    out_class_list = [self.idx_dict.dict_modified[pred_cls.item()] for pred_cls in predicted_class_tensor]
                    print("out_class_tensor : ", out_class_list, len(out_class_list))
                    heatmap_top_3_list = []
                    class_top_3_list = []
                    for idx in range(len(out_class_list)):
                        grad_cam_heatmap = self.gradcam.show_cam_on_image(img=img_np, mask=heatmap_tensor[idx], use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.5)
                        predicted_class = out_class_list[idx]
                        heatmap_top_3_list.append(grad_cam_heatmap)
                        class_top_3_list.append(predicted_class)
                    result_dict["grad_cam_heatmap_top-3"].append(heatmap_top_3_list)
                    result_dict["out_class_top-3"] += class_top_3_list
                    result_dict["grad_cam_arr"] = np.stack(result_dict["grad_cam_heatmap_top-3"])
                    result_list.append(result_dict)



                else:
                    heatmap, predicted_class = self.gradcam.run_grad_cam(img_tensor=img_tensor, img_np=img_np, img_class=270)
                    # 이 부분 수정하기 : mode = "top-3"일때와 "default"일때 heatmap, predicted_class 차원 차이 확인 ..print(heatmap.shape)
                    print("predicted_class : ", predicted_class, predicted_class.shape)
                    out_class = [self.idx_dict.dict_modified[pred_cls.item()] for pred_cls in predicted_class]
                    print("out_class : ", out_class, len(out_class))
                    grad_cam_heatmap = self.gradcam.show_cam_on_image(img=img_np, mask=heatmap, use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.5)
                    print(f"grad_cam_heatmap shape = {grad_cam_heatmap.shape}")
                    result_dict["grad_cam_heatmap"].append(grad_cam_heatmap)
                    result_dict["out_class"] += out_class
                    result_dict["grad_cam_arr"] = np.stack(result_dict["grad_cam_heatmap"])
                    result_list.append(result_dict)

        return result_list

    def gradcam_heatmap_visualize(self, result_list, save_plot_dir=None):
        if self.gradcam.mode == "top-3":
            for idx in range(len(result_list)):
                img_path = result_list[idx]["img_path"]
                out_class_top_3 = result_list[idx]["out_class_top-3"]
                grad_cam_arr_top_3 = result_list[idx]["grad_cam_arr"]

                filename_with_ext = os.path.basename(img_path)
                filename, _ = os.path.splitext(filename_with_ext)

                img_original = Image.open(img_path).convert('RGB')
                out_series_top_3 = pd.Series(out_class_top_3).value_counts()
                out_df = pd.DataFrame(out_series_top_3, columns=["count"], index=out_series_top_3.index).reset_index()
                predicted_label_top_3 = pd.Series(out_class_top_3).value_counts().index[0]

                grad_cam_heatmap_top_3 = np.squeeze(grad_cam_arr_top_3, axis=0)

                for i in range(len(out_class_top_3)):
                    grad_cam_heatmap = grad_cam_heatmap_top_3[i]
                    predicted_label = out_class_top_3[i]
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
                    save_filename = os.path.join(save_plot_dir, f"{predicted_label}_{filename}.png")
                    plt.savefig(save_filename, bbox_inches='tight')
                    plt.close(fig)

        else:
            for idx in range(len(result_list)):
                img_path = result_list[idx]["img_path"]
                out_class = result_list[idx]["out_class"]
                grad_cam_arr = result_list[idx]["grad_cam_arr"]

                filename_with_ext = os.path.basename(img_path)
                filename, _ = os.path.splitext(filename_with_ext)

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
                save_filename = os.path.join(save_plot_dir, f"{predicted_label}_{filename}.png")
                plt.savefig(save_filename, bbox_inches='tight')
                plt.close(fig)