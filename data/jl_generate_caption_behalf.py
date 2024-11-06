import pandas as pd
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import os
from tqdm import tqdm
import open_clip

input_path = "/data1/jianglei/coser_imagenet-1K"  # path for HR

name_list = []
# half=504904
half=10
num = 0
with open(f"data/ImageNet/Obj512_all/all.txt", 'r') as f:
    for line in f.readlines():
        if num < half: 
            name_list.append(line.rstrip('\n'))
            num += 1
image_path_num = len(name_list)        

result_name = f"data/ImageNet/Obj512_all/blip2_imagenet_captions_all_behalf.json"

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
torch_dtype = torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"

# device_ids = [2, 3]  
# model = torch.nn.DataParallel(model, device_ids=device_ids)
# if torch.cuda.device_count() > 1:
#     print("Using multiple GPUs...")
# model = torch.nn.DataParallel(model)
model.to(device, dtype=torch.float16)

clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', device=torch.device("cuda"))
tokenizer = open_clip.get_tokenizer('ViT-H-14')

df = pd.DataFrame(columns=['filename', 'caption1', 'caption2', 'caption3', 'clip_score1', 'clip_score2', 'clip_score3'], dtype=float)

batch_size = 256  # Define batch size for processing
num_images = len(name_list)

for i in tqdm(range(0, num_images, batch_size)):
    batch_names = name_list[i:i + batch_size]
    images = [Image.open(os.path.join(input_path, name)).convert('RGB') for name in batch_names]
    
    processor.padding_side='left'
    # Preprocess images and create a batch
    inputs = processor(images, return_tensors="pt",padding=True).to(device, torch.float16)

    # Generate captions for the entire batch
    generated_ids =model.generate(**inputs, max_new_tokens=20)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    captions1 = [text.lower().replace('.', ',').rstrip(',\n') for text in generated_texts]

    # Prepare batch inputs for the second prompt (a photo of ...)
    prompt = "a photo of"
    inputs_with_prompt = processor(images, text=[prompt] * len(images), return_tensors="pt", padding=True).to(device, torch.float16)

    # Generate captions for the entire batch with the prompt
    generated_ids_with_prompt =model.generate(**inputs_with_prompt, max_new_tokens=20)
    generated_texts_with_prompt = processor.batch_decode(generated_ids_with_prompt, skip_special_tokens=True)
    captions2 = [prompt + ' ' + text.lower().replace('.', ',').rstrip(',\n') for text in generated_texts_with_prompt]

    # Prepare batch inputs for the third prompt (question-based prompt)
    prompt = "Question: Please describe the contents in the photo in details. Answer:"
    inputs_with_question = processor(images, text=[prompt] * len(images), return_tensors="pt", padding=True).to(device, torch.float16)

    # Generate captions for the entire batch with the question prompt
    generated_ids_with_question =model.generate(**inputs_with_question, max_new_tokens=20)
    generated_texts_with_question = processor.batch_decode(generated_ids_with_question, skip_special_tokens=True)
    captions3 = [text.lower().replace('.', ',').rstrip(',\n') for text in generated_texts_with_question]

    # Calculate CLIP scores for the entire batch
    # image_tensors = torch.stack([preprocess(image).half().to(device) for image in images])
    # text_tensors = tokenizer(captions1 + captions2 + captions3).to(device)
    text_probs = []
    for j in range(len(captions1)):
        image = preprocess(images[j]).unsqueeze(0)
        text = tokenizer([captions1[j], captions2[j], captions3[j]])
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = clip_model.encode_image(image.half().cuda())
            text_features = clip_model.encode_text(text.cuda())
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_prob = image_features @ text_features.T
            text_probs.append(text_prob)

    for j in range(len(batch_names)):
        df.loc[len(df.index)] = [batch_names[j], captions1[j], captions2[j], captions3[j], 
                                 float(text_probs[j][0,0]), float(text_probs[j][0, 1]), float(text_probs[j][0, 2])]
        print( [batch_names[j], captions1[j], captions2[j], captions3[j], 
                                 float(text_probs[j][ 0,0]), float(text_probs[j][0, 1]), float(text_probs[j][0, 2])])

df.to_json(result_name)
