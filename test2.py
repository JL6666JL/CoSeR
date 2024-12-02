import torch
from torchvision.transforms import ToPILImage
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter, Decoder, DALLE2
from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig, TrainDecoderConfig


prior_config = TrainDiffusionPriorConfig.from_json_path("content/prior_config.json").prior
prior = prior_config.create().cuda()

prior_model_state = torch.load("content/prior_weights.pth")
prior.load_state_dict(prior_model_state, strict=True)

decoder_config = TrainDecoderConfig.from_json_path("content/decoder_config.json").decoder
decoder = decoder_config.create().cuda()

decoder_model_state = torch.load("content/decoder_weights.pth")["model"]

for k in decoder.clip.state_dict().keys():
    decoder_model_state["clip." + k] = decoder.clip.state_dict()[k]

decoder.load_state_dict(decoder_model_state, strict=True)

dalle2 = DALLE2(prior=prior, decoder=decoder).cuda()

images = dalle2(
    ['a dog playing a ball'],
    cond_scale = 2.
).cpu()

print(images.shape)

img = ToPILImage()(images[0])
img.save("image.png")
