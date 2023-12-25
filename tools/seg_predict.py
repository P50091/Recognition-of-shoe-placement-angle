from ultralytics import YOLO
import os
import cv2
import numpy as np
from tqdm import tqdm

def segment_images(image_folder, model_path,output_folder_mask):
    # 创建输出文件夹
    os.makedirs(output_folder_mask, exist_ok=True)

    # 加载预训练模型
    model = YOLO(model_path)
    # 读取文件夹中的所有图像
    for filename in tqdm(os.listdir(image_folder), desc="Processing images"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(image_folder, filename)

            # 对图像进行推理
            results = model(img_path,imgsz=1024)
            for result in results:
                # 读取原始图像
                original_img = cv2.imread(img_path)

                # 创建一个全零数组
                mask_tmp = np.zeros((1024, 1024, 3))

                # 对每个对象进行迭代
                for i in range(len(result.masks)):
                    mask = result.masks.data[i].cpu().numpy()  # 获取对象的分割蒙版
                    mask = np.expand_dims(mask, axis=0)
                    # 复制通道维度
                    mask = np.transpose(mask, (1, 2, 0))
                    mask = np.dstack([mask, mask, mask])
                    # 累加 mask
                    mask_tmp += mask

                # 保存蒙版图像
                mask_output_path = os.path.join( output_folder_mask, f'{filename}')
                cv2.imwrite(mask_output_path, mask_tmp * 255)  # 将蒙版值从[0,1]转换为[0,255]

# 使用函数
if __name__ == '__main__':
    segment_images('../data/images', '../weight/best_seg.pt', "../data/mask")
