import io
import math
import os
import random
import sys
import time
from datetime import datetime
import numpy as np
import torch
import imageio.v3 as iio

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import skimage
import skvideo
import skvideo.datasets
import skvideo.io
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import os
import re
import subprocess
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F

from helpers import (calculate_moments, correlation, get_mgrid, print_moments,
                     print_top_k_freq_moments, print_top_k_frequencies, psnr,
                     read, top_k_frequencies, write)
from SIREN import dataio

random.seed(4)
torch.manual_seed(4)

device = torch.device("cuda:0")

### HYPERPARAMETERS ###
media_type = "video" # supports ["image", "gradients", "audio", "video"]
object_of_the_file = sys.argv[1] # Comment this out unless running from cmd
version = sys.argv[2] # Comment this out unless running from cmd
max_chunk_size = 1 * 40000 # TODO only set this

def activation_func(x):
    return x
    return torch.sin(x) * torch.sqrt(torch.maximum(torch.abs(x), torch.tensor(1e-4, requires_grad=True)))

'''
Usage: 

CUDA_VISIBLE_DEVICES=0 python3 SPDER_video_batched.py bikes SPDER
'''

test_SIREN = False
test_PE = True
num_hidden_layers = 12
num_neurons = 1024

# SELECT WHICH CODE PATH TO RUN
collect_data = True

omega_0 = 30 # the 'w' in sin(wx) * sqrt(|x|). Pretty much the only thing you can "tune" in the activation

if version == "SPDER":
    def activation_func(x):
        return torch.sin(x) * torch.sqrt(torch.maximum(torch.abs(x), torch.tensor(1e-43, requires_grad=True)))
    test_PE = False
    test_SIREN = False
elif version == "SIREN":
    test_PE = True
    test_SIREN = True
elif version == "RELU":
    activation_func = lambda x: torch.relu(x)
    test_PE = False
    test_SIREN = False
elif version == "RELU_PE":
    activation_func = lambda x: torch.relu(x)
    test_PE = True
    test_SIREN = False
# SELECT WHICH CODE PATH TO RUN

total_steps = 5001

show_plt_plots = False
show_super_res = False
resolution = 256 # for show_super_res
show_activation_distribution = False

image_base_length = 1024
learning_rate = 5e-6
sample_rate_for_video = 0.25 / 100
steps_til_summary = 25
### HYPERPARAMETERS ###

num_hidden_layers = 12
num_neurons = 1024
learning_rate = 5e-6

def largest_factor_less_than(a, b):
    factor = b
    while factor > 0:
        if a % factor == 0: # and factor % 8 == 0:
            return factor
        factor -= 1
    return factor

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

# This is basically a measure of how much the gradient changed in a given direction
def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)

class FourierFeatureEncodingPositional(nn.Module):
    '''Module to add fourier features as in Tancik[2020].'''

    def __init__(self, in_features, num_frequencies=20, scale=1):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.frequencies = scale ** (torch.range(0, num_frequencies - 1) / num_frequencies)
        self.frequencies = self.frequencies.cuda()
        self.scale = scale
        self.out_dim = 2 * in_features * self.num_frequencies

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)
        coord_freq = torch.einsum('p, sqr->sqrp', self.frequencies, coords)
        sin = torch.sin(2 * np.pi * coord_freq)
        cos = torch.cos(2 * np.pi * coord_freq)
        coords_pos_enc = torch.cat((sin, cos), axis=-1)
        res = coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)
        return res

class SPDERLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=omega_0, is_first=False, 
    activation_func=lambda x: torch.sin(x) * torch.sqrt(torch.abs(x))):
        super().__init__()        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.omega_0 = omega_0
        self.is_first = is_first
        self.activation_func = activation_func
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)     
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        # TODO change "intermediate" to "x" since it's more intuitive for me
        intermediate = self.omega_0 * self.linear(input)
        if test_SIREN == True:
            return torch.sin(intermediate)
        else:
            return self.activation_func(intermediate)

class SPDER_Network(nn.Module):
    def __init__(self, in_features, num_neurons, hidden_layers, out_features, activation_func):
        super().__init__()

        if test_PE:
            self.positional_encoding = PosEncodingNeRF(in_features=in_features, sidelength=image_base_length)
            in_features = self.positional_encoding.out_dim
        
        self.net = []

        # Input layer
        self.net.append(SPDERLayer(in_features, num_neurons, activation_func=activation_func, 
                                  omega_0=omega_0, is_first=True))

        # Hidden layers
        for i in range(hidden_layers):
            self.net.append(SPDERLayer(num_neurons, num_neurons, activation_func=activation_func, 
                                      omega_0=omega_0, is_first=False))

        # Output layer
        final_linear = nn.Linear(num_neurons, out_features)

        with torch.no_grad():
            # todo how does this work?
            final_linear.weight.uniform_(-np.sqrt(6 / num_neurons) / omega_0, 
                                            np.sqrt(6 / num_neurons) / omega_0)

        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

        self.intermediate_outputs = []

    def forward(self, coords):
        # coords are a [1, 256^2, 2] dimension tensor representing x, y pairs scaled from -1 to 1
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input

        # The reason to detach the tensor from its computation history
        #  and then to compute its gradients is to allow the input
        #  tensor to be modified during the forward pass without
        #  affecting the computation history of other parts of the
        #  network. By detaching the tensor, we make sure that the
        #  gradients of the input tensor will not propagate backwards
        #  through the computation history, which could have unintended
        #  consequences.

        if test_PE:
            coords = self.positional_encoding(coords)

        output = coords
        self.intermediate_outputs = []

        for layer in self.net:
            output = layer(output)
            self.intermediate_outputs.append(output)

        return output, coords

def regularized_weights_loss(model, l_norm=1, l_lambda=0.001):
    l_norm = sum(torch.norm(param, l_norm) for param in model.parameters())
    return l_lambda * l_norm

def get_image_object_tensor(sidelength_x):
    # if image_file is default, go to the skimage.data.camera()
    # TODO allow for any image that i feed in as image_file to be used
    # for the camera, the image is an nd_array of size 512 by 512, with whole numbers from 0 to 255 in each entry
    image_file = object_of_the_file
    if image_file == "default" or image_file == 'camera':
        nd_array = skimage.data.camera()
        img = Image.fromarray(nd_array)
    elif image_file == "gigapixel" or image_file == 'girl':
        img = Image.open("data/21girl.jpg")
    elif image_file == "albert":
        exr_file = "data/albert.exr"
        tensor_data = exr_to_tensor(exr_file)
        numpy_data = tensor_data.numpy()
        img = Image.fromarray((numpy_data * 255).astype(np.uint8))  
    elif image_file == "astronaut":
        nd_array = skimage.data.astronaut()
        img = Image.fromarray(nd_array)
    elif image_file == "eagle":
        nd_array = skimage.data.eagle()
        img = Image.fromarray(nd_array)
    elif image_file == "cat":
        nd_array = skimage.data.cat()
        img = Image.fromarray(nd_array)
    elif image_file == "random":
        mean = 128
        stddev = 60
        rand_array = np.clip(np.round(np.random.normal(mean, stddev, (256, 256))), 0, 255).astype(np.uint8)
        nd_array = rand_array
        img = Image.fromarray(nd_array)
    else:
        # Open the input image
        img = Image.open(image_file)

    # Get the width and height of the image
    width, height = img.size

    # Determine the size of the square crop
    crop_size = min(width, height)

    # Calculate the top-left corner of the crop
    left = 0
    top = 0

    # Crop the image to a square
    img = img.crop((left, top, left+crop_size, top+crop_size))

    print("Using image file", image_file)

    transform = Compose([
        Resize(sidelength_x), # reduces to 256 by 256 by taking average of nearby pixels
        ToTensor(), # converts to tensor

        # todo consider removing / tweaking this
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])) # normalizes all the pixel values by (x-0.5)/0.5 i dunno y
        # todo consider removing / tweaking this
        ])

    img_tensor = transform(img)
    # of size [1, 256, 256]
    print("The img tensor has shape", img_tensor.shape)
    return img_tensor

def flatten_intermediate_outputs(int_outputs):
        output = []
        for layer in int_outputs:
            output += layer.flatten().tolist()

        return output 

class ImageFitting(Dataset):
    '''
    This gives us the image ready to be fed in with its coordinates and corresponding pixel colors
    It takes the tensor object and preprocesses it for DataLoader
    '''
    def __init__(self, sidelength):
        super().__init__()
        img = get_image_object_tensor(sidelength)

        self.pixels = img.permute(1, 2, 0) # now of torch size [256, 256, 1] H W C 
        
        try:
            self.pixels = self.pixels.view(-1, 1) # vectorizes two dimensional pixel matrix with 1 channel
            print("self.pixels using grayscale")
        except RuntimeError:
            self.pixels = self.pixels.view(-1, 3)
            print("self.pixels using R G B")

        self.coords = get_mgrid(sidelength, 2) 
    
    def __len__(self):
            return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels

# What media input goes into the model?
if media_type == "video":
    if object_of_the_file == "bunny":
        video_path = "data/bunny.mp4"
    elif object_of_the_file == "bikes":
        video_path = skvideo.datasets.bikes()
    else:
        video_path = object_of_the_file
    assert type(video_path) is str

    print("VIDEO PATH", video_path)
    # Video object is nd array w attribute shape (num_frames, height, width, num_channels)
    # and attribute channels for number of channels (should be 3)
    vid_dataset = dataio.Video(video_path)
    print("vid_dataset shape", vid_dataset.shape, vid_dataset.channels)

    f, h, w = vid_dataset.shape
    elements_in_video_tensor = int(f * h * w * vid_dataset.channels)
    
    # coord_dataset is linearized to tensor of shape (x, 3)
    # x is height * width * frames
    # and then everytime you sample, it picks sampling_rate proportion of any of these linearized ones
    coord_dataset = dataio.Implicit3DWrapper(vid_dataset, sidelength=vid_dataset.shape, sample_fraction=1)

    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
    in_features = 3
    out_features = 1
else:
    raise NotImplementedError

SPDER_trained_model = SPDER_Network(in_features=in_features, 
                        out_features=out_features, 
                        num_neurons=num_neurons, 
                        hidden_layers=num_hidden_layers,
                        activation_func=activation_func)

print("Using", torch.cuda.device_count(), "GPUs")
SPDER_trained_model = torch.nn.DataParallel(SPDER_trained_model)

SPDER_trained_model = SPDER_trained_model.to(device)
    
# Load previous model
# SPDER_trained_model.module.load_state_dict(torch.load('serialized/bikes_video5.0e-06SPDER.pth'))

print(type(SPDER_trained_model))
print(SPDER_trained_model.state_dict().keys())

optim = torch.optim.Adam(lr=learning_rate, params=SPDER_trained_model.parameters())

if media_type == "video":
    model_input, ground_truth = next(iter(dataloader))
    model_input = model_input["coords"][:, 0:, 0:3] # weird formatting for output of dataloader
    ground_truth = ground_truth["img"][:, 0:, 0:1]
    
else:
    raise NotImplementedError

print("model input", type(model_input), (model_input.shape if type(model_input) != dict else model_input.keys()), model_input,)
print("ground_truth", type(ground_truth), (ground_truth.shape if type(ground_truth) != dict else ground_truth.keys()), ground_truth)

learning_rate_str = "{:.1e}".format(learning_rate)
losses = []
activation_name = object_of_the_file + "_" + media_type + str(learning_rate_str) 
time_measurement = time.time()

activation_values = []

model_input = model_input.to(device)
ground_truth = ground_truth.to(device)

lowest_loss = float("inf")

total_elements = len(ground_truth.flatten().tolist())

chunk_size = largest_factor_less_than(total_elements, max_chunk_size)
num_gpus = torch.cuda.device_count()

assert chunk_size % num_gpus  == 0, "Chunk size must be a multiple of num_gpus for it to split evenly amongst gpus"
print("Chunk size", chunk_size, "num_gpus", num_gpus, "total_elements", total_elements)
assert total_elements % chunk_size == 0, "Total elements must be a multiple of chunk size"

num_chunks = total_elements // chunk_size
print("Using batch size", chunk_size, "with total elements", total_elements, "and num chunks", num_chunks)

for step in range(total_steps):
    optim.zero_grad()
    loss = 0
    
    chunk_start_time = time.time()
    for chunk_num in range(num_chunks):
        
        start = chunk_num * chunk_size
        end = (chunk_num + 1) * chunk_size
        
        model_input_batch = model_input[:, start:end, :]
        ground_truth_batch = ground_truth[:, start:end, :]
        
        model_input_batch = model_input_batch.view(num_gpus, -1, in_features)
        ground_truth_batch = ground_truth_batch.view(num_gpus, -1, out_features)
        
        model_prediction, coords = SPDER_trained_model(model_input_batch)  
        
        assert model_prediction.shape == ground_truth_batch.shape
        miniloss = ((model_prediction - ground_truth_batch) ** 2).mean() 
        miniloss /= num_chunks
        miniloss.backward()
        
        loss += miniloss.item()
        
        
        if chunk_num % 10 == 0:
            chunk_end_time = time.time()
            print("Step", step, "Chunk", chunk_num, "Loss estimate", miniloss.item() * num_chunks, "Took", chunk_end_time - chunk_start_time, "seconds")
            chunk_start_time = chunk_end_time
            
    loss = torch.Tensor([loss])
    lowest_loss = min(lowest_loss, float(loss.item()))
    if np.isnan(loss.item()):
        print("Got nan loss")
        exit()
        
    write("ground_truth_" + os.path.basename(object_of_the_file) + "_" + media_type, ground_truth)
    write("model_input_" + os.path.basename(object_of_the_file) + "_" + media_type, model_input)
    
    print("Step %d, Lowest loss %0.10f" % (step, lowest_loss), "Took", round(time.time() - time_measurement, 2), "seconds")
    
    time_measurement = time.time()

    optim.step()
    losses.append(loss.item())
    
    if test_SIREN:
        write(activation_name + version + "_SIREN_losses.file", losses)
        torch.save(SPDER_trained_model.module.state_dict(), "serialized/" + activation_name + version + "_SIREN.pth")
    elif test_PE:
        write(activation_name + version + "_PE_losses.file", losses)
        torch.save(SPDER_trained_model.module.state_dict(), "serialized/" + activation_name + version + "_PE.pth")
    else:
        torch.save(SPDER_trained_model.module.state_dict(), "serialized/" + activation_name + version + ".pth")
        write(activation_name + version + "_losses.file", losses)
    
    if collect_data:
        model_pred_to_use = model_prediction.detach().cpu()
        good_step = False
        dataset_dir = "serialized/" + media_type + "datasetdata/"

        if media_type == "video":
            write(activation_name + version + "_losses.file", losses)
