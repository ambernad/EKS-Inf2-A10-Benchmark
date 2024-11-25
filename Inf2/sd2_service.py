from flask import Flask, request
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

app = Flask(__name__)
pipe = None

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

# Load the models and any necessary dependencies
def load_models():
    global pipe
    print("1")
    COMPILER_WORKDIR_ROOT = '/opt/sd2_compile_dir_512'
    model_id = "stabilityai/stable-diffusion-2-1-base"
    text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
    decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
    unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
    post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')
    print("2")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    print("3")
    # Load the compiled UNet onto two neuron cores.
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    device_ids = [0,1]
    pipe.unet.unetwrap = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)
    print("4")
    # Load other compiled models onto a single neuron core.
    pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
    pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
    pipe.vae.decoder = torch.jit.load(decoder_filename)
    pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
    print("5")
    pass

# API endpoint for performing inference
@app.route('/inference', methods=['POST'])
def inference():
    global pipe
    prompt = request.get_json()['prompt']
    total_time = 0
    start_time = time.time()
    image = pipe(prompt).images[0]
    total_time = time.time() - start_time

    # Generate image key with current timestamp up to milliseconds
    current_time = time.strftime("%Y%m%d%H%M%S%f")[:-3]
    image_key = f"generated_image_{current_time}.png"
    bucket_name = "abernads-sd2-images"

    # Save image to local file
    image_path = "image.png"
    image.save(image_path)

    # Upload image to S3 bucket
    s3 = boto3.client('s3')
    s3.upload_file(image_path, bucket_name, image_key)
    
    result = "Inference time: " + str(np.round(total_time, 2)) + ". Image saved to S3 bucket " + bucket_name + " with key " + image_key 
    return {'result': result}

# Start the Flask application
if __name__ == '__main__':
    load_models()  # Load models on startup
    app.run(host='0.0.0.0')