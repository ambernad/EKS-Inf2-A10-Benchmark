from flask import Flask, request
import os

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import time

import copy
import boto3
import sys

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.cross_attention import CrossAttention

app = Flask(__name__)
pipe = None

# Load the models and any necessary dependencies
def load_models():
    global pipe
    model_id = "stabilityai/stable-diffusion-2-1-base"

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
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
    image_key = f"gpu_generated_image_{current_time}.png"
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