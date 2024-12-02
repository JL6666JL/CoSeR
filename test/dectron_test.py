# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

im = cv2.imread("/home/jianglei/work/CoSeR/test/ball_dog.png")
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)
print(outputs.keys())
print(outputs['panoptic_seg'][0].shape)
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# print((out.get_image()[:, :, ::-1].shape))
# cv2.imshow("pic",out.get_image()[:, :, ::-1])
cv2.imwrite('dectron.png',out.get_image()[:, :, ::-1])

print(outputs['panoptic_seg'][1])
print(outputs['instances'])
predicted_labels = np.argmax(outputs['sem_seg'].cpu(),axis=0)

H,W = predicted_labels.shape
# color_map = {
#     1: [255, 0, 0],  
#     2: [0, 255, 0],  
#     3: [0, 0, 255], 
#     4: [255, 255, 0]   
# }


color_map = {num: np.random.rand(3)*255 for num in range(55)}
color_map[0]=np.zeros(3)
rgb_image = np.zeros((H, W, 3))
for value, color in color_map.items():
    rgb_image[ outputs['panoptic_seg'][0].cpu() == value] = color

# np.savetxt('tensor.txt', rgb_image, fmt='%d')  
# target_color = np.array([0, 0, 0], dtype=np.uint8)
# for i in range(H):
#     for j in range(W):
#         print(rgb_image[i][j])

cv2.imwrite('seg_pic.png',rgb_image)

# H,W = outputs['panoptic_seg'][0].shape
# color_map = {
#     1: [255, 0, 0],  
#     2: [0, 255, 0],  
#     3: [0, 0, 255], 
#     4: [255, 255, 0]   
# }
# rgb_image = np.zeros((H, W, 3))
# for value, color in color_map.items():
#     rgb_image[ outputs['panoptic_seg'][0].cpu() == value] = color

# np.savetxt('tensor.txt', outputs['sem_seg'].cpu(), fmt='%d')  
# # target_color = np.array([0, 0, 0], dtype=np.uint8)
# # for i in range(H):
# #     for j in range(W):
# #         if np.all(rgb_image[i][j] == target_color):
# #             input('wrong!')

# cv2.imwrite('seg_pic.png',rgb_image)
