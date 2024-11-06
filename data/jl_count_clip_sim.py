from basicsr.archs.vgg_arch import VGGFeatureExtractor
import os
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from torch import nn as nn
import pandas as pd
import pickle
import open_clip
from torch.utils.data import Dataset, DataLoader

file_path = "/data1/jianglei/coser_imagenet-1K_new"
result_name = "data/ImageNet/Obj512_all/imagenet_all_clipcls.pkl"
feature_name = "data/ImageNet/Obj512_all/imagenet_all_clipcls_feature.npy"

layer_weights = {'conv1_2': 0.1, 'conv2_2': 0.1, 'conv3_4': 1, 'conv4_4': 1, 'conv5_4': 1}
model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', device=torch.device("cuda"))
tokenizer = open_clip.get_tokenizer('ViT-H-14')

class ImageDataset(Dataset):
    def __init__(self, file_path, img_list):
        self.file_path = file_path
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.file_path, self.img_list[idx])
        image = preprocess(Image.open(img_path))
        return img_path.split("/")[-1], image

# Load image paths
paths = []
with open('data/ImageNet/Obj512_all/all.txt') as fin:
    for line in fin.readlines():
        paths.append(line.rstrip('\n'))

# Create dataset and dataloader
dataset = ImageDataset(file_path, paths)
batch_size = 32  # Adjust the batch size as needed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=4)

feature_list = []
for filenames, images in tqdm(dataloader):
    images = images.half().cuda()  # Move images to GPU and convert to half precision
    with torch.no_grad(), torch.cuda.amp.autocast():
        features = model.encode_image(images)
    features = features.cpu().detach().numpy()
    
    for filename, feature in zip(filenames, features):
        classname = filename.split("_")[0]
        feature_list.append([filename, classname, feature])

# Save features to a DataFrame and then to a NumPy file
feature_df = pd.DataFrame(feature_list)
feature_df.set_index([1], inplace=True)
feature_list_numpy = np.array(feature_df[2].tolist())
np.save(feature_name, feature_list_numpy)

# similarity
saved_feature = np.load(feature_name)
feature_list = []
# feature_list_numpy = []
for i in range(len(paths)):
    path = paths[i]
    filename = path.split("/")[-1]
    classname = filename.split("_")[0]
    feature = saved_feature[i]
    feature_list.append([filename, classname, feature])

feature_df = pd.DataFrame(feature_list)
feature_df.set_index([1], inplace=True)


df = pd.read_table('data/ImageNet/class.txt', sep='\t', header=None)
class_list = list(df[1])

sim_dict = {}
for class_name in tqdm(class_list):
    if class_name in feature_df.index:
        class_feature_df = feature_df.loc[class_name]
        class_image_names = list(class_feature_df[0])
        
        class_feature_all = torch.Tensor(np.array(class_feature_df[2].tolist())).to("cuda", torch.float32)
        class_feature_all /= class_feature_all.norm(dim=-1, keepdim=True)
        class_loss = class_feature_all @ class_feature_all.T

        sim_dict[class_name] = {'filename': class_image_names, 'loss': class_loss.cpu().detach().numpy()}

with open(result_name, 'wb') as tf:
    pickle.dump(sim_dict, tf)
