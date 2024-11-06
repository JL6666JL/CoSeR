## 使用nms将一张图片中的不同物体分割出来，并且保存为单独的图片
import torch
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from torchvision.ops import nms
from PIL import Image

# 1. 初始化模型
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置预测阈值
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# 2. 读取图片并进行分割
image = cv2.imread("/home/jianglei/work/CoSeR/test/ball_dog.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
outputs = predictor(image)

# 3. 获取分割掩码、边界框和置信度
masks = outputs["instances"].pred_masks.cpu().numpy()
boxes = outputs["instances"].pred_boxes.tensor.cpu()
scores = outputs["instances"].scores.cpu()

# 4. 应用非极大值抑制，设定重叠阈值
iou_threshold = 0.7  # IOU 阈值（重叠度）
keep = nms(boxes, scores, iou_threshold)  # 返回保留的框索引

# 5. 保留 NMS 后的分割掩码和边界框
selected_masks = masks[keep.numpy()]
selected_boxes = boxes[keep].numpy()

# 6. 提取每个分割区域并保存
output_size = (128, 128)  # 统一大小，例如 128x128
for i, mask in enumerate(selected_masks):
    segmented_img = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
    segmented_img[mask] = np.concatenate([image_rgb[mask], np.full((mask.sum(), 1), 255)], axis=1)
    
    x1, y1, x2, y2 = selected_boxes[i].astype(int)
    cropped_img = segmented_img[y1:y2, x1:x2]
    resized_img = cv2.resize(cropped_img, output_size)
    
    # 保存图像
    Image.fromarray(resized_img).save(f"segmented_object_{i}.png")
