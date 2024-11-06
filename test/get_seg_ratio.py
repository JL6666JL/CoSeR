import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.detection_utils import read_image
import os
from tqdm import tqdm  # 导入 tqdm

# 配置模型
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设定阈值
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

def is_image_segmentable(image_path):
    """检测图像是否有实例分割结果"""
    image = read_image(image_path, format="BGR")
    outputs = predictor(image)
    # 检查是否存在实例分割结果
    if "instances" in outputs and len(outputs["instances"]) > 0:
        return True
    return False

def calculate_segmentable_ratio(dataset_dir):
    """计算数据集中可分割图像的比例"""
    image_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith((".jpg", ".png",".JPEG"))]
    segmentable_count = 0
    
    # 使用 tqdm 创建进度条
    for image_path in tqdm(image_files, desc="Processing images"):
        if is_image_segmentable(image_path):
            segmentable_count += 1
    
    total_images = len(image_files)
    if total_images == 0:
        return 0
    return segmentable_count / total_images

# 示例用法
dataset_dir = "/data1/jianglei/coser_imagenet-1K_new"  # 替换为数据集路径
output_file = "segmentable_ratio.txt"  # 输出文件名

ratio = calculate_segmentable_ratio(dataset_dir)

# 将结果保存到文件中
with open(output_file, "w") as f:
    f.write(f"可分割图像的比例: {ratio * 100:.2f}%\n")
print(f"可分割图像的比例: {ratio * 100:.2f}%\n")

print(f"结果已保存到 {output_file}")
