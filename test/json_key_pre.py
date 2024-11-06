# import json

# # 读取原始 JSON 文件
# with open('test_be.json', 'r') as file:
#     data = json.load(file)

# # 创建一个新的字典，用于存储修改后的数据
# new_data = {}

# # 遍历原始数据并更新 key
# for main_key, sub_dict in data.items():
#     new_data[main_key] = {str(int(sub_key) + 10): value for sub_key, value in sub_dict.items()}

# # 保存为没有换行符的 JSON 文件
# with open('test_af.json', 'w') as file:
#     json.dump(new_data, file, indent=None, separators=(',', ':'))

#############################################################################
# import json
# from tqdm import tqdm

# # 读取原始 JSON 文件
# with open('/home/jianglei/work/CoSeR/data/ImageNet/Obj512_all/blip2_imagenet_captions_all_afhalf.json', 'r') as file:
#     data = json.load(file)

# # 创建一个新的字典，用于存储修改后的数据
# new_data = {}
# be_num = 504904

# # 遍历原始数据并更新 key
# for main_key, sub_dict in tqdm(data.items()):
#     new_data[main_key] = {str(int(sub_key) + be_num): value for sub_key, value in sub_dict.items()}

# # 保存为没有换行符的 JSON 文件
# with open('blip2_imagenet_captions_all_afhalf.json', 'w') as file:
#     json.dump(new_data, file, indent=None, separators=(',', ':'))

#############################################################################
import json

# 读取原始 JSON 文件
with open('/home/jianglei/work/CoSeR/merged_file.json', 'r') as file:
    data = json.load(file)

flag = 0

# 遍历原始数据并更新 key
for main_key, sub_dict in data.items():
    cur = -1
    for sub_key,value in sub_dict.items():
        if int(sub_key) - cur != 1:
            flag = 1
            print(int(sub_key) - cur,sub_key)
        cur = int(sub_key)
if flag:
    print('wrong')
else:
    print('right')

