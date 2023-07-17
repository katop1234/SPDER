import os
import pickle
import random
import time
from base64 import b64encode

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.stats as statistics
import skimage
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import wavfile
from torch import nn

HOME_DIR = os.getcwd()

print("HOME DIR", HOME_DIR)

activation_func = lambda: None

def correlation(x, y):
    #assert len(x) == len(y)
    numerator = sum([x[i] * y[i] for i in range(len(x))])
    denom_sqrt_1 = sum([x[i] ** 2 for i in range(len(x))]) ** 0.5
    denom_sqrt_2 = sum([y[i] ** 2 for i in range(len(y))]) ** 0.5   

    out = numerator / (denom_sqrt_1 * denom_sqrt_2)
    
    if not(-1 <= out <= 1):
        print("GOT A CORRELATION GREATER THAN 1, SOMETHNG'S FUNKY", out)
        exit()
    return out


def calculate_moments(l):
    
    mean = np.mean(l)
    variance = np.var(l) 
    skewness = statistics.skew(l)
    kurtosis = statistics.kurtosis(l)
    return mean, variance, skewness, kurtosis


def print_moments(l):
    mean, variance, skewness, kurtosis = calculate_moments(l)
    print("Mean:", round(mean, 6))
    print("Variance:", round(variance, 6))
    print("Skewness:", round(skewness, 6))
    print("Kurtosis:", round(kurtosis, 6))
    return mean, variance, skewness, kurtosis


def inner_product(v1, v2):
    assert type(v1) == list and type(v2) == list
    length1 = len(v1)
    length2 = len(v2)
    min_length = min(length1, length2)
    inner_product = 0
    for i in range(min_length):
        inner_product += v1[i] * v2[i]
    return inner_product


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    return dot_product / (magnitude1 * magnitude2)


# Get the frequency representations of all 3
def fft_amps(arr):
    if not type(arr) is list:
        arr = arr.detach().numpy()
    amps = np.fft.fft(arr)
    amps = np.squeeze(amps).flatten()
    return [np.abs(x) for x in amps]

# Get the frequency representations of all 3
def fft_phase(arr):
    phases = np.fft.fft(arr.detach().numpy())
    phases = np.squeeze(phases).flatten()
    return [np.angle(x) for x in phases]

def print_top_k_frequencies(arr, k=30):
    fft = np.fft.fft(arr)
    amplitudes = np.abs(fft)
    sorted_indices = np.argsort(amplitudes)[::-1]
    top_k = sorted_indices[:k]
    for index in top_k:
        print(f"Frequency: {index}, Amplitude: {amplitudes[index]}")

def video_fft(video):
    # Compute the FFT of each frame in the video
    fft_frames = np.fft.fftn(video, axes=(0, 1))

    # Shift the FFT to center the zero frequency component
    fft_shifted = np.fft.fftshift(fft_frames, axes=(0, 1))

    # Compute the magnitude spectrum for each frame
    mag_spectrum = 20 * np.log(np.abs(fft_shifted))

    return mag_spectrum

def top_k_frequencies(arr, k=30):
    fft = np.fft.fft(arr)
    amplitudes = np.abs(fft)
    sorted_indices = np.argsort(amplitudes)[::-1]
    top_k = sorted_indices[:k]
    sorted_amplitudes = np.sort(amplitudes[top_k])[::-1]
    return sorted_amplitudes

def print_top_k_freq_moments(arr, k=30):
    list1 = fft_amps(arr)
    list2 = np.fft.fftfreq(len(arr), 1/100)

    assert len(list2) == len(list1)
    combined_list = list(zip(list1, list2))
    sorted_list = sorted(combined_list, key=lambda x: x[0], reverse=True)
    amps = []
    freqs = []
    for i in range(k):
        amps.append(sorted_list[i][0])
        freqs.append(sorted_list[i][1])
    moments = print_moments(freqs)
    print("")
    return moments

def get_abs_from_2d_fft(arr):
    return [np.abs(x) for x in arr]

def get_phase_from_2d_fft(arr):
    return [np.angle(x) for x in arr]

def run_fft2d(arr):
    complex_arr = arr.to(torch.complex64)
    result = torch.fft.fft2(complex_arr)
    result = result.squeeze().tolist()
    return get_abs_from_2d_fft(result), get_phase_from_2d_fft(result)

def l2_loss(list1, list2):
    # Convert lists to numpy arrays
    arr1 = np.array(list1)
    arr2 = np.array(list2)

    # Calculate the difference between the arrays
    diff = arr1 - arr2

    # Calculate the L2 loss (sum of squared differences)
    loss = np.sum(diff ** 2)

    return loss

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def make_square(arr):
    # Find the minimum dimension of the array
    min_side = min(arr.shape)

    # Calculate the size of the square
    square_size = (min_side, min_side)

    # Create a new square array with the minimum size
    square_arr = np.zeros(square_size, dtype=arr.dtype)

    # Copy the contents of the original array into the square array
    square_arr[:min_side, :min_side] = arr[:min_side, :min_side]

    return square_arr

def inner_product(v1, v2):
    assert type(v1) == list and type(v2) == list
    length1 = len(v1)
    length2 = len(v2)
    min_length = min(length1, length2)
    inner_product = 0
    for i in range(min_length):
        inner_product += v1[i] * v2[i]
    return inner_product

def get_home_dir():
    return HOME_DIR

def get_serialized_dir():
    curr = os.getcwd()
    os.chdir(get_home_dir())
    output = os.path.join(os.getcwd(), "serialized/")
    os.chdir(curr)
    return output

import os
import pickle

def psnr(img1, img2):
    img1 = (255/2) * (img1 + 1)
    img2 = (255/2) * (img2 + 1)

    mse = torch.mean((img1 - img2) ** 2)
    max_pixel = 255
    return 20 * torch.log10(max_pixel / mse)

def write(name, obj, versioned=False):
    serialized_dir = get_serialized_dir()
    
    if name.startswith("serialized/"):
        name = name[len("serialized/"):]

    filename = os.path.join(serialized_dir, name)

    if versioned:
        version = 2
        while os.path.exists(filename):
            filename = os.path.join(serialized_dir, f"{name}.{version}")
            version += 1
            if version > 9:
                raise Exception("Too many versions of the file already exist")

    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read(name):
    curr_dir = os.getcwd()
    os.chdir(get_serialized_dir())

    if name.startswith("serialized/"):
        name = name[len("serialized/"):]

    with open(name, 'rb') as f:
        obj = pickle.load(f)

    os.chdir(curr_dir)
    return obj

def initialize_superres_lists_old():
    datasets = ["Set5", "Set14", "sunhays80", "bsd100", "urban100"]
    superresolution_factors = [2, 3, 4]

    for dataset in datasets:
        for srf in superresolution_factors:
            filename = os.path.join('serialized', 'superres_data', f'{dataset}_SRF_{srf}_losses')
            write(filename, [])

def initialize_dataset_lists(media):
    print("WARNING, ABOUT TO DELETE ALL LISTS OF DATA FOR", media)
    time.sleep(1)
    print("LAST CALL")
    time.sleep(1)

    if media == "image":
        numbers = [100, 400, 800, 25, 100, 500]
        versions = ['SPDER', 'SIREN', 'RELU', 'RELU_PE', "RELU_FFN"]
        metrics = ['losses', 'psnrs', 'corrs']
    elif media == "video":
        numbers = [100, 500]
        versions = ['SPDER', 'SIREN', 'RELU', 'RELU_PE']
        metrics = ['losses', 'psnrs', 'corrs']
    elif media == "audio":
        numbers = [25, 100, 500, 1000]
        versions = ['SPDER', 'SIREN', 'RELU', 'RELU_PE', "RELU_FFN"]
        metrics = ['losses', 'corrs']
        
    folder = f'serialized/{media}datasetdata/'
    # Loop through the files and delete them
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        os.remove(file_path)

    for version in versions:
        for metric in metrics:
            for num in numbers:
                print(version, metric, num)
                filename = f'serialized/{media}datasetdata/{version}_{num}_step_{metric}'
                # Use helpers.write("serialized/audiodatasetdata/RELU_FFN_25_step_corrs", [])
                
                write(filename, [])
                print("replaced", filename)
    
    assert read(filename) == []
    print("Done")

def play_mp4(filename):
    html = ''
    video = open(filename,'rb').read()
    src = 'data:video/mp4;base64,' + b64encode(video).decode()
    html += '<video width=1000 controls autoplay loop><source src="%s" type="video/mp4"></video>' % src 
    return HTML(html)

def delete_files_in_dir(dir_path="serialized/"):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def get_fourier_series(values):
    return np.fft.fft(values)

class AudioFile(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.rate, self.data = wavfile.read(filename)
        self.data = self.data.astype(np.float32)
        self.timepoints = get_mgrid(len(self.data), 1)

    def get_num_samples(self):
        return self.timepoints.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        amplitude = self.data
        scale = np.max(np.abs(amplitude))
        amplitude = (amplitude / scale)
        amplitude = torch.Tensor(amplitude).view(-1, 1)
        return self.timepoints, amplitude

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        # for weather only
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        intermediate = self.omega_0 * self.linear(input)

        return torch.sin(intermediate)
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)

        return torch.sin(intermediate), intermediate

def initialize_superres_data_lists():
    # Define possible options
    datasets = ['DIV2K', 'Flickr2K_Train', 'ffhq0', 'ffhq1', 'ffhq2']
    model_types = ['SIREN', 'SPDER', 'Bicubic']
    superres_options = [(512, 1024), (512, 2048), (512, 4096)]

    # Iterate over all combinations
    for dataset in datasets:
        for model_type in model_types:
            for superres in superres_options:
                # Define paths for loss and PSNR files
                losses_path = os.path.join('superres_data', f'{dataset}_{model_type}_{superres}_losses')
                psnrs_path = os.path.join('superres_data', f'{dataset}_{model_type}_{superres}_psnrs')
                corrs_path = os.path.join('superres_data', f'{dataset}_{model_type}_{superres}_corrs')

                # Initialize empty lists and write to file
                write(losses_path, [])
                write(psnrs_path, [])
                write(corrs_path, [])

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

class SPDERLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30, is_first=False):
        super().__init__()
        
        self.in_features = in_features
        
        self.linear = nn.Linear(in_features, out_features, bias=True)
        
        self.omega_0 = omega_0
        self.is_first = is_first
        
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
        return torch.sin(intermediate) * torch.sqrt(torch.abs(intermediate))

class SPDER_Network(nn.Module):
    def __init__(self, in_features, nums_neurons, hidden_layers, out_features, omega_0=30):
        super().__init__()
        self.net = []

        # Input layer
        self.net.append(SPDERLayer(in_features, nums_neurons, 
                                  omega_0=omega_0, is_first=True))
        
        # Hidden layers
        for i in range(hidden_layers):
            self.net.append(SPDERLayer(nums_neurons, nums_neurons, 
                                      omega_0=omega_0, is_first=False))

        # Output layer
        final_linear = nn.Linear(nums_neurons, out_features)
            
        with torch.no_grad():
            # todo how does this work?
            final_linear.weight.uniform_(-np.sqrt(6 / nums_neurons) / 30, 
                                            np.sqrt(6 / nums_neurons) / 30)
            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
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
        
        output = self.net(coords)
        return output, coords  

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

