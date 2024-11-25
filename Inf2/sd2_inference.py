import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

import torch
import torch.nn as nn
import torch_neuronx
import numpy as np

import time
import copy
import boto3
import sys

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.cross_attention import CrossAttention

class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        out_tuple = self.unet(sample, timestep, encoder_hidden_states, return_dict=False)
        return out_tuple

class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states, cross_attention_kwargs=None):
        sample = self.unetwrap(sample, timestep.float().expand((sample.shape[0],)), encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)

class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]

def run_inference_and_save_image(prompt, bucket_name, image_key):
    COMPILER_WORKDIR_ROOT = '/opt/sd2_compile_dir_512'
    model_id = "stabilityai/stable-diffusion-2-1-base"
    text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
    decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
    unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
    post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Load the compiled UNet onto two neuron cores.
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    device_ids = [0,1]
    pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)

    # Load other compiled models onto a single neuron core.
    pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
    pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
    pipe.vae.decoder = torch.jit.load(decoder_filename)
    pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)

    total_time = 0
    start_time = time.time()
    image = pipe(prompt).images[0]
    total_time = total_time + (time.time() - start_time)

    # Generate image key with current timestamp up to milliseconds
    current_time = time.strftime("%Y%m%d%H%M%S%f")[:-3]
    image_key = f"generated_image_{current_time}.png"

    # Save image to local file
    image_path = "image.png"
    image.save(image_path)

    # Upload image to S3 bucket
    s3 = boto3.client('s3')
    s3.upload_file(image_path, bucket_name, image_key)

    return total_time, image_key

# Main function
def main():
    # Read prompt from command-line argument or use default prompt
    prompt = sys.argv[1] if len(sys.argv) > 1 else "fox jump over frog"

    # Set S3 bucket name and image key
    bucket_name = "abernads-sd2-images"
    image_key = ""

    # Run inference and save image to S3 bucket
    elapsed_time, image_key = run_inference_and_save_image(prompt, bucket_name, image_key)
    print("Inference time:", np.round(elapsed_time, 2), "seconds")
    print("Image saved to S3 bucket:", bucket_name, "with key:", image_key)

if __name__ == "__main__":
    main()
