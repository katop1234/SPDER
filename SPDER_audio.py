import io
import math
import os
import random
import sys
import time
from datetime import datetime

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import skimage
import skvideo
import skvideo.datasets
import skvideo.io
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from helpers import (calculate_moments, correlation, get_mgrid, print_moments,
                     print_top_k_freq_moments, print_top_k_frequencies, psnr,
                     read, top_k_frequencies, video_fft, write)
from SIREN import dataio, diff_operators

### HYPERPARAMETERS ###
media_type = "audio" # supports ["image", "gradients", "audio", "video", "sdf"]
object_of_the_file = sys.argv[1] # Comment this out unless running from cmd
version = sys.argv[2] # Comment this out unless running from cmd

test_SIREN = False
test_PE = False
test_FFN = False
num_hidden_layers = 5
num_neurons = 256

# SELECT WHICH CODE PATH TO RUN
collect_data = False
activation_func = lambda: None

random.seed(4)
torch.manual_seed(4)

if version == "SPDER":
    def activation_func(x):
        #return torch.sin(x) * torch.sqrt(torch.maximum(torch.abs(x), torch.tensor(1e-43, requires_grad=True)))
        return torch.sin(x) * torch.arctan(x)
    test_PE = False
    test_SIREN = False
elif version == "SIREN":
    test_PE = False
    test_SIREN = True
    def activation_func(x):
        return torch.sin(x)
elif version == "RELU":
    activation_func = lambda x: torch.relu(x)
    test_PE = False
    test_SIREN = False
elif version == "RELU_PE":
    activation_func = lambda x: torch.relu(x)
    test_PE = True
elif version == "RELU_FFN":
    activation_func = lambda x: torch.relu(x)
    test_FFN = True
# SELECT WHICH CODE PATH TO RUN

omega_0 = 30 # the 'w' in sin(wx) * sqrt(|x|). Pretty much the only thing you can "tune" in the activation
total_steps = 5001

'''
PE + ReLU (NERF)
SIREN
SPDER
ReLU
'''

show_plt_plots = False
show_super_res = False
resolution = 256 # for show_super_res
show_activation_distribution = False

image_base_length = 256
learning_rate = 5e-5
sample_rate_for_video = 0.25 / 100
steps_til_summary = 25
audio_length = 308200
batch_size = 77050
### HYPERPARAMETERS ###

assert not (test_PE and test_FFN)
batch_size = min(batch_size, audio_length)

if audio_length % batch_size != 0:
    accum_iters = audio_length // batch_size + 1
    # +1 for anything left over
else:
    accum_iters = audio_length // batch_size
    # otherwise it divides it nicely

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
            fn_samples = audio_length
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
        elif test_FFN:
            self.positional_encoding = FourierFeatureEncodingPositional(in_features=in_features)
        
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


# What media input goes into the model?
if media_type == "audio":
    if object_of_the_file == "bach":
        audio_dataset = dataio.AudioFile(filename="data/gt_bach.wav")
    else:
        audio_dataset = dataio.AudioFile(filename=object_of_the_file)
    coord_dataset = dataio.ImplicitAudioWrapper(audio_dataset)
    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
    in_features = 1
    out_features = 1

else:
    raise NotImplementedError

SPDER_trained_model = SPDER_Network(in_features=in_features, 
                        out_features=out_features, 
                        num_neurons=num_neurons, 
                        hidden_layers=num_hidden_layers,
                        activation_func=activation_func)

SPDER_trained_model = SPDER_trained_model.cuda()
optim = torch.optim.Adam(lr=learning_rate, params=SPDER_trained_model.parameters())

if media_type == "image":
    model_input, ground_truth = next(iter(dataloader))
    ground_truth = ground_truth[:, :, 0:1]
elif media_type == "audio":
    model_input, ground_truth = next(iter(dataloader))
    model_input = model_input["coords"]
    ground_truth = ground_truth["func"][0].view(1, -1, 1)
    
    model_input = model_input[:, :audio_length, :]
    ground_truth = ground_truth[:, :audio_length, :]

elif media_type == "gradients":
    model_input, ground_truth = next(iter(dataloader))
    model_input = model_input["coords"] # weird formatting for output of dataloader
    ground_truth = ground_truth["gradients"]
elif media_type == "video":
    model_input, ground_truth = next(iter(dataloader))
    model_input = model_input["coords"] # weird formatting for output of dataloader
    ground_truth = ground_truth["img"]
elif media_type == "sdf":
    model_input, ground_truth = next(iter(dataloader))
    model_input = model_input["coords"]
else:
    raise NotImplementedError

print("model input", type(model_input), (model_input.shape if type(model_input) != dict else model_input.keys()), model_input,)
print("ground_truth", type(ground_truth), (ground_truth.shape if type(ground_truth) != dict else ground_truth.keys()), ground_truth)

learning_rate_str = "{:.1e}".format(learning_rate)
losses = []
activation_name = object_of_the_file + "_" + media_type + str(learning_rate_str) 
time_measurement = time.time()

activation_values = []

lowest_loss = float("inf")
best_corr = -1
for step in range(total_steps):
    optim.zero_grad()
    loss = torch.Tensor([0])
    
    model_preds = []
    
    for iter in range(accum_iters):
        start_idx = iter * batch_size
        end_idx = min((iter + 1) * batch_size, audio_length)
        
        batched_model_input = model_input[:, start_idx:end_idx, :].cuda()
        batched_ground_truth = ground_truth[:, start_idx:end_idx, :].cuda()
        
        batched_model_prediction, coords = SPDER_trained_model(batched_model_input) 
        
        batched_model_pred_to_add = batched_model_prediction.detach().cpu().flatten()
        model_preds.append(batched_model_pred_to_add)
        
        loss_measurement = ((batched_model_prediction - batched_ground_truth) ** 2).mean()
        loss_measurement /= ((end_idx - start_idx) / audio_length)
        loss_measurement.backward()
        
        loss += loss_measurement.detach().cpu()

        assert batched_model_prediction.shape == batched_ground_truth.shape
        del batched_model_prediction, batched_ground_truth, batched_model_input
    
    if np.isnan(loss.item()):
        print("Got nan loss")
        exit()
    
    if not collect_data:
        write("ground_truth_" + os.path.basename(object_of_the_file) + "_" + media_type, ground_truth)
        write("model_input_" + os.path.basename(object_of_the_file) + "_" + media_type, model_input)

    print("Step %d, Total loss %0.10f" % (step, loss), "Lowest loss", lowest_loss, "Took", round(time.time() - time_measurement, 2), "seconds")
    time_measurement = time.time()

    optim.step()
    losses.append(loss.item())
    lowest_loss = min(lowest_loss, float(loss.item()))
    
    model_prediction = []
    for tensor in model_preds:
        model_prediction.extend(tensor.tolist())
    
    model_prediction = torch.Tensor(model_prediction).cpu()
    model_pred_to_use = model_prediction.detach().cpu()
    
    rate = audio_dataset[0][0]
    
    ground_truth_to_write = ground_truth * coord_dataset.scale
    model_pred_to_use_to_write = model_pred_to_use * coord_dataset.scale
    
    model_pred_for_fft = model_pred_to_use.view(-1, 1).cpu()
    ground_truth_for_fft = ground_truth.view(-1, 1).cpu()
    print("model_pred_to_use", model_pred_to_use.shape)
    print("ground_truth", ground_truth.shape)
    a = np.abs(np.fft.fft(model_pred_for_fft.flatten().tolist()))
    b = np.abs(np.fft.fft(ground_truth_for_fft.flatten().tolist()))

    assert len(a) == len(b)

    # Calculate the correlation between the two arrays using the cross-correlation method
    corr = correlation(a, b)
    
    print("Correlation", corr)
    best_corr = max(best_corr, corr)

    if not collect_data:
        # Commented out writing the model because it slows down training by a lot
        if test_SIREN:
            write(activation_name + version + "_SIREN_losses.file", losses)
            #write(activation_name + version + "_SIREN_model.file", SPDER_trained_model)
        elif test_PE:
            write(activation_name + version + "_PE_losses.file", losses)
            #write(activation_name + version + "_PE_model.file", SPDER_trained_model)
        else:
            write(activation_name + version + "_losses.file", losses)
            #write(activation_name + version + "_model.file", SPDER_trained_model)
            
    elif collect_data:

        good_step = False
        dataset_dir = "serialized/" + media_type + "datasetdata/"
        if media_type == "image":
            if step in [25, 100, 500]:
                loss_file_path = dataset_dir + version + f"_{step}_step_losses"
                psnr_file_path = dataset_dir + version + f"_{step}_step_psnrs"
                corr_file_path = dataset_dir + version + f"_{step}_step_corrs"
                good_step = True
                pass 

        elif media_type == "audio":
            if step in [25, 100, 500, 1000]:
                loss_file_path = dataset_dir + version + f"_{step}_step_losses"
                corr_file_path = dataset_dir + version + f"_{step}_step_corrs"
                good_step = True
                pass 
        
        if good_step == True:
            # Write the loss
            serialized_losses = read(loss_file_path)
            serialized_losses.append(lowest_loss)
            write(loss_file_path, serialized_losses)

            if media_type == "image":
                # Write the PSNR
                model_pred_image = model_pred_to_use.view(image_base_length, image_base_length).cpu()
                ground_truth_rescaled_image = ground_truth.view(image_base_length, image_base_length).cpu()
                psnr_val = psnr(model_pred_image, ground_truth_rescaled_image)
                serialized_psnrs = read(psnr_file_path)
                serialized_psnrs.append(psnr_val)
                write(psnr_file_path, serialized_psnrs)

                # Write the amplitude correlation
                # Get two 2D arrays
                a = np.abs(np.fft.fft2(model_pred_to_use)).flatten().tolist()
                b = np.abs(np.fft.fft2(ground_truth.detach().cpu())).flatten().tolist()

                assert len(a) == len(b)

                # Calculate the correlation between the two arrays using the cross-correlation method
                corr = correlation(a, b)
                serialized_corrs = read(corr_file_path)
                serialized_corrs.append(corr)
                write(corr_file_path, serialized_corrs)
            
            elif media_type == "video":
                # Write the psnr value
                
                # Write the amplitude correlation
                # Get two 2D arrays
                a = np.abs(video_fft(model_pred_to_use)).flatten().tolist()
                b = np.abs(video_fft(ground_truth)).flatten().tolist()

                assert len(a) == len(b)

                # Calculate the correlation between the two arrays using the cross-correlation method
                corr = correlation(a, b)
                serialized_corrs = read(corr_file_path)
                serialized_corrs.append(corr)
                write(corr_file_path, serialized_corrs)
                pass
            
            elif media_type == "audio":
                ground_truth = ground_truth.detach().cpu()
                # Write the amplitude correlation
                # Get two 2D arrays
                corr = best_corr

                assert len(a) == len(b)

                # Calculate the correlation between the two arrays using the cross-correlation method
                serialized_corrs = read(corr_file_path)
                serialized_corrs.append(corr)
                write(corr_file_path, serialized_corrs)
                psnr_val = None
                psnr_file_path = None
                pass

            print("Wrote corr", corr, corr_file_path, "loss", lowest_loss, loss_file_path, "psnr", psnr_val, psnr_file_path, "on object", object_of_the_file)
