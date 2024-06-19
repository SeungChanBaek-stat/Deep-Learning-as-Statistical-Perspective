from PIL import Image
import numpy as np
import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os



class MCGradCAMVisualize:
    def __init__(self, mcgradcam, idx_dict, img_list, bs, T):
        self.mcgradcam = mcgradcam
        self.idx_dict = idx_dict.dict_modified
        self.mcd_result_list = []
        self.img_list = img_list
        self.bs = bs
        self.T = T

    def calc_cls_entropy(self, prob):
        return -(prob * np.log(prob)).sum()

    def mcgradcam_heatmap_calc(self):
        img_list = self.img_list
        mcd_result_list = self.mcd_result_list
        for i, img_path_ in enumerate(img_list, 1):
            mcd_result_dict = {}
            img_tensor = self.mcgradcam.load_img(img_path=img_path_)
            mcd_result_dict["img_path"] = img_path_
            mcd_result_dict["img_tensor"] = img_tensor.cpu()
            heatmap, predicted_class, img_np = self.mcgradcam.run_mc_grad_cam(img_path=img_path_, nmc=self.T, bs= self.bs)
            mcd_result_dict["heatmap"] = heatmap.detach().cpu().numpy()
            mcd_result_dict["predicted_class"] = predicted_class.detach().cpu().numpy()
            mcd_result_dict["img_np"] = img_np

            out_class = mcd_result_dict["predicted_class"]
            mc_heatmap = mcd_result_dict["heatmap"]

            out_series = pd.Series(out_class).value_counts()
            out_df = pd.DataFrame(out_series, columns=["count"], index=out_series.index).reset_index()
            out_df["index"] = [self.idx_dict[out_ind] for out_ind in out_df["index"]]
            out_df.index = out_df["index"]
            predicted_label = self.idx_dict[pd.Series(out_class).value_counts().index[0]]
            out_df["prob"] = out_df["count"] / out_df["count"].sum()

            mcd_result_dict["out_df"] = out_df
            mcd_result_dict["cls_entropy"] = self.calc_cls_entropy(out_df["prob"].values)

            # Initialize a dictionary to hold heatmaps for each class
            heatmaps_by_class = {class_name: [] for class_name in out_df['index']}

            # Group heatmaps by predicted class
            for pred_class, heatmap in zip(out_class, mc_heatmap):
                class_name = self.idx_dict[pred_class]
                if class_name in heatmaps_by_class:
                    heatmaps_by_class[class_name].append(heatmap)
                else:
                    heatmaps_by_class[class_name] = [heatmap]

            mcd_result_dict["heatmaps_by_class_list"] = heatmaps_by_class

            mcd_result_list.append(mcd_result_dict)

            # del img_tensor, heatmap, predicted_class, img_np, mc_heatmap
            # gc.collect()
            # # torch.cuda.empty_cache()
            print(f"{i}번째 사진 완료")
        print(f"mcd_result_list 갯수 = {len(mcd_result_list)}")
        print(f"mcd_result_list[0] keys = {mcd_result_list[0].keys()}")
        print(f"mcd_result_list[0]['out_df'] shape = {mcd_result_list[0]['out_df'].shape}")
        print(f"mcd_result_list[0]['heatmaps_by_class_list'] shape = {mcd_result_list[0]['heatmaps_by_class_list'].shape}")
        print(f"mcd_result_list[0]['heatmaps'] keys = {mcd_result_list[0]['heatmaps'].shape}")

        return mcd_result_list

    def mcgradcam_heatmap_visualize(self, result_list, save_plot_dir=None):
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