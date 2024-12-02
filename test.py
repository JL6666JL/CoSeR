import torch
from PIL import Image
import open_clip
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork,  OpenAIClipAdapter
from dalle2_pytorch.train_configs import TrainDecoderConfig, TrainDiffusionPriorConfig, DecoderConfig, UnetConfig, DiffusionPriorConfig

model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
model.to('cuda:0')
tokenizer = open_clip.get_tokenizer('ViT-L-14')

image = preprocess(Image.open("test/ball_dog.png")).unsqueeze(0).to('cuda:0')
text = tokenizer(["a diagram", "a dog with a ball", "a cat"]).to('cuda:0')

prior_config = TrainDiffusionPriorConfig.from_json_path("content/prior_config.json").prior
prior = prior_config.create().cuda()
prior_model_state = torch.load("content/prior_best.pth")
prior.load_state_dict(prior_model_state, strict=True)


with torch.no_grad(), torch.cuda.amp.autocast():
    # image_features = model.encode_image(image)
    # text_features = model.encode_text(text)
    image_features, _ = prior.clip.embed_image(image)
    text_features = prior.sample(text, num_samples_per_batch = 2, cond_scale = 10)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
