import torch
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from torchvision.ops import nms
from PIL import Image

# COCO类别名称列表
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
    "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


# 1. 初始化模型
cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置预测阈值
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置预测阈值
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# 2. 读取图片并进行分割
image = cv2.imread("/home/jianglei/work/CoSeR/test/n01537544_31.JPEG")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
outputs = predictor(image)

# 3. 获取分割掩码、边界框、置信度和类别
masks = outputs["instances"].pred_masks.cpu().numpy()
boxes = outputs["instances"].pred_boxes.tensor.cpu()
scores = outputs["instances"].scores.cpu()
classes = outputs["instances"].pred_classes.cpu().numpy()  # 类别索引

# 仅保留置信度大于 0.7 的检测结果
confidence_threshold = 0.7
high_conf_indices = scores > confidence_threshold
masks = masks[high_conf_indices.numpy()]
boxes = boxes[high_conf_indices]
scores = scores[high_conf_indices]
classes = classes[high_conf_indices.numpy()]

# 4. 应用非极大值抑制，设定重叠阈值
iou_threshold = 0.7  # IOU 阈值（重叠度）
keep = nms(boxes, scores, iou_threshold)  # 返回保留的框索引

# 5. 保留 NMS 后的分割掩码和边界框
selected_masks = masks[keep.numpy()]
selected_boxes = boxes[keep].numpy()
selected_classes = classes[keep.numpy()]

# 6. 提取每个分割区域并保存
output_size = (512, 512)  # 统一大小
for i, mask in enumerate(selected_masks):
    # 创建一个黑色的图像（只包含 RGB 三通道）
    segmented_img = np.zeros_like(image_rgb, dtype=np.uint8)
    
    # 在分割区域内保留图像内容，其他区域为黑色
    segmented_img[mask] = image_rgb[mask]
    
    x1, y1, x2, y2 = selected_boxes[i].astype(int)
    cropped_img = segmented_img[y1:y2, x1:x2]
    resized_img = cv2.resize(cropped_img, output_size)
    
    # 获取类别名称
    class_name = COCO_CLASSES[selected_classes[i]]
    
    # 保存图像，文件名包含类别名称
    Image.fromarray(resized_img).save(f"segmented_object_{class_name}_{i}.JPEG")
