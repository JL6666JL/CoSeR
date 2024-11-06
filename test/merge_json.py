import json

file1 = "/home/jianglei/work/CoSeR/data/ImageNet/Obj512_all/blip2_imagenet_captions_all_behalf.json"
file2 = "/home/jianglei/work/CoSeR/data/ImageNet/Obj512_all/blip2_imagenet_captions_all_afhalf.json"
# 加载两个 JSON 文件
with open(file1) as f1, open(file2) as f2:
    json1 = json.load(f1)
    json2 = json.load(f2)

# 遍历 json2，将其中的键值对合并到 json1
for key in json1:
    if key in json2:
        json1[key].update(json2[key])

# 保存合并后的 JSON 文件
with open('merged_file.json', 'w') as outfile:
    json.dump(json1, outfile, indent=4)
