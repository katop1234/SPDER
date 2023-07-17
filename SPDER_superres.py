
'''
Use this script to collect data on superresolution
'''

import re, os, sys
import glob
import torch
import torch
import torch.nn.functional as F
import torchvision
from helpers import read, write, initialize_superres_data_lists, torch, get_mgrid, correlation
from torch import nn
import skimage
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import numpy as np 
import time

### Hyperparameters ###
# Check if command line arguments were provided, else use default
dataset = sys.argv[1] if len(sys.argv) > 1 else "DIV2K" # Valid options: DIV2K, Flickr2K_Train, ffhq0, ffhq1, ffhq2
version = sys.argv[2] if len(sys.argv) > 2 else "SPDER" # Valid options: SPDER, SIREN
LR_sidelength_to_use = int(sys.argv[3]) if len(sys.argv) > 3 else 512
SRF_factors = [2, 4, 8]

print("Using args: ", dataset, version, LR_sidelength_to_use, SRF_factors)

data_folder = ...
num_gpus = torch.cuda.device_count()
max_chunk_size = num_gpus * 100000 # 60000 is the largest chunk size that fits on my 3090
omega_0 = 30 # the 'w' in sin(wx) * sqrt(|x|). Pretty much the only thing you can "tune" in the activation
test_SIREN = False
test_PE = False
test_FFN = False
num_hidden_layers = 5
num_neurons = 256
num_channels = 1
total_steps = 101 # TODO change
media_type = "image"
### Hyperparameters ###

if version == "SPDER" or version == "spder":
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

### Helpers ###
def img_num_exists(img_num):
    img_num_str = str(img_num).zfill(3) # pad with leading zeros to get 3 digits
    matching_files = glob.glob(f'img_{img_num_str}*')
    return len(matching_files) > 0

def get_lr_img_file_path(curr_img_num, srf):
    img_num_str = str(curr_img_num).zfill(3) # pad with leading zeros to get 3 digits
    matching_files = glob.glob(f'img_{img_num_str}_SRF_{srf}_LR.png')
    if len(matching_files) > 0:
        return matching_files[0]
    else:
        return None
    
def get_hr_img_file_path(curr_img_num, srf):
    img_num_str = str(curr_img_num).zfill(3) # pad with leading zeros to get 3 digits
    matching_files = glob.glob(f'img_{img_num_str}_SRF_{srf}_HR.png')
    if len(matching_files) > 0:
        return matching_files[0]
    else:
        return None

def get_shorter_sidelength(HR_img_file_path):
    with Image.open(HR_img_file_path) as img:
        return min(img.size)

def get_SRF_losses_list(SRF, dataset):
    return os.path.join("superres_data", f'{dataset}_SRF_{SRF}_losses')

def largest_factor_less_than(a, b):
    factor = b
    while factor > 0:
        if a % factor == 0: # and factor % 8 == 0:
            return factor
        factor -= 1
    return factor

def img_tensor_to_uint8(tensor):
    return 255 * (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))

def get_MSE(gt, pred_img):

    assert gt.shape == pred_img.shape, "Ground truth and prediction images must have the same shape."

    # Normalize the images
    gt = img_tensor_to_uint8(gt) / (255 / 2) - 1
    pred_img = img_tensor_to_uint8(pred_img) / (255 / 2) - 1

    # Compute and return MSE
    mse = torch.mean((gt - pred_img) ** 2)
    return mse

@torch.no_grad()
def forward_pass_in_chunks(model, tensor, max_chunk_size=max_chunk_size):
    # Calculate the number of chunks we'll need
    num_chunks = (tensor.shape[0] + max_chunk_size - 1) // max_chunk_size

    # Initialize a list to hold our results
    results = []

    # Iterate over each chunk
    for i in range(num_chunks):
        # Calculate the start and end indices for this chunk
        start_index = i * max_chunk_size
        end_index = min((i + 1) * max_chunk_size, tensor.shape[0])

        # Get the chunk from the tensor
        chunk = tensor[start_index:end_index]

        # Run the model on this chunk and get the result
        result, _ = model(chunk)

        # Append the result to our results list
        results.append(result)

    # Concatenate the results along the appropriate dimension
    output = torch.cat(results, dim=0)

    # Return the concatenated output
    return output

### Helpers ###

def activation_func(x):
    return x
    return torch.sin(x) * torch.sqrt(torch.maximum(torch.abs(x), torch.tensor(1e-4, requires_grad=True)))

# SELECT WHICH CODE PATH TO RUN
collect_data = False

omega_0 = 30 # the 'w' in sin(wx) * sqrt(|x|). Pretty much the only thing you can "tune" in the activation

if version == "SPDER":
    def activation_func(x):
        return torch.sin(x) * torch.sqrt(torch.maximum(torch.abs(x), torch.tensor(1e-43, requires_grad=True)))
    test_PE = False
    test_SIREN = False
elif version == "SIREN":
    test_PE = False
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

show_plt_plots = False
show_super_res = False
show_activation_distribution = False

learning_rate = 1e-4
### HYPERPARAMETERS ###

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

def get_PSNR(tensor1, tensor2):
    # Flatten tensors
    tensor1 = tensor1.reshape(-1)
    tensor2 = tensor2.reshape(-1)

    # Ensure tensors have the same size
    assert tensor1.shape == tensor2.shape, "Tensors must have the same size"

    # Normalize tensors to be between -1 and 1
    tensor1 = 2 * (tensor1 - tensor1.min()) / (tensor1.max() - tensor1.min()) - 1
    tensor2 = 2 * (tensor2 - tensor2.min()) / (tensor2.max() - tensor2.min()) - 1

    # Compute MSE
    mse = F.mse_loss(tensor1, tensor2)

    # Compute PSNR
    psnr = 10 * torch.log10(4 / mse)

    return psnr.item()

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
            self.positional_encoding = PosEncodingNeRF(in_features=in_features, sidelength=256)
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
        #  consequences

        if test_PE:
            coords = self.positional_encoding(coords)

        output = coords
        self.intermediate_outputs = []

        for layer in self.net:
            output = layer(output)
            self.intermediate_outputs.append(output)

        return output, coords
    
def save_tensor_as_image(tensor, file_name):
    n = int(tensor.shape[0] ** 0.5)
    assert n * n == tensor.shape[0], "Tensor's first dimension should be a perfect square"

    # Scale tensor from its min and max to 0 and 255
    tensor = ((tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor)) * 255).round().byte()

    # Reshape tensor to square image
    tensor = tensor.view(n, n, -1)  # -1 will automatically size the channel dimension

    # Transpose tensor to meet image format [height, width, channels]
    tensor = tensor.permute(1, 0, 2)

    # Convert tensor to numpy array and then to PIL image
    if tensor.shape[-1] == 1:  # grayscale image
        img = Image.fromarray(tensor.cpu().numpy().squeeze(), 'L')  # 'L' mode for grayscale
    elif tensor.shape[-1] == 3:  # RGB image
        img = Image.fromarray(tensor.cpu().numpy(), 'RGB')
    else:
        raise ValueError("Unsupported number of channels. The number of channels must be either 1 (grayscale) or 3 (RGB).")

    # Save image
    img.save(file_name)

def get_image_object_tensor(sidelength_x, object_of_the_file):
    # if image_file is default, go to the skimage.data.camera()
    # TODO allow for any image that i feed in as image_file to be used
    # for the camera, the image is an nd_array of size 512 by 512, with whole numbers from 0 to 255 in each entry
    image_file = object_of_the_file
    if image_file == "default" or image_file == 'camera':
        nd_array = skimage.data.camera()
        img = Image.fromarray(nd_array)

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
        img = Image.fromarray(np.round(np.array(img).astype(np.float32) * 255 / 255).astype(np.uint8))
        
    # Get the width and height of the image
    width, height = img.size

    # Determine the size of the square crop
    crop_size = min(width, height)

    # Calculate the top-left corner of the crop
    left = 0
    top = 0

    # Crop the image to a square
    img = img.crop((left, top, left+crop_size, top+crop_size))

    print("Using image file", image_file, "in", os.getcwd())

    transform = Compose([
        Resize(sidelength_x), # reduces to 256 by 256 by taking average of nearby pixels
        ToTensor(), # converts to tensor

        # todo consider removing / tweaking this
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])) # normalizes all the pixel values by (x-0.5)/0.5 i dunno y
        # todo consider removing / tweaking this
        ])

    img_tensor = transform(img).float()
    
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
    def __init__(self, sidelength, object_of_the_file):
        super().__init__()
        img = get_image_object_tensor(sidelength, object_of_the_file)

        self.pixels = img.permute(1, 2, 0) # now of torch size [256, 256, 1] H W C 
        
        if self.pixels.shape[-1] == 1:
            self.pixels = self.pixels.view(-1, 1) # vectorizes two dimensional pixel matrix with 1 channel
        elif self.pixels.shape[-1] == 3:
            self.pixels = self.pixels.view(-1, 3)
        else:
            raise NotImplementedError
            
        self.coords = get_mgrid(sidelength, 2) 
    
    def __len__(self):
            return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels

from scipy import ndimage

def bicubic_interpolation(input_image, target_size):
    # Read the image using PIL
    image = Image.open(input_image)
    
    # Convert to numpy array
    image_np = np.array(image)

    # Scale factors
    scale_x = target_size / image_np.shape[1]
    scale_y = target_size / image_np.shape[0]

    # Perform bicubic interpolation
    bicubic_img = ndimage.zoom(image_np, (scale_y, scale_x, 1), order=3)

    # Convert to PyTorch tensor and return
    return torch.from_numpy(bicubic_img).float().permute(2, 0, 1).cuda()

def train_network(image_path):
                  
    # What media input goes into the model?
    if media_type == "image":
        image_base_length = LR_sidelength_to_use
        image_object = ImageFitting(image_base_length, image_path)
        model_input_object = image_object
        dataloader = DataLoader(model_input_object, batch_size=1, pin_memory=True, num_workers=0)
        in_features = 2 # x, y
        
        # num_channels = get_image_object_tensor(image_base_length, image_path).shape[0]      
        out_features = num_channels
    else:
        raise NotImplementedError
    
    model = SPDER_Network(in_features=2, 
                        out_features=num_channels, 
                        num_neurons=num_neurons, 
                        hidden_layers=num_hidden_layers,
                        activation_func=activation_func) # TODO set activation function based on version if not already
    
    model = torch.nn.DataParallel(model)

    model = model.cuda()

    optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())

    if media_type == "image":
        model_input, ground_truth = next(iter(dataloader))
        ground_truth = ground_truth[:, :, 0:num_channels].cuda()
    else:
        raise NotImplementedError

    time_measurement = time.time()

    model_input = model_input.cuda()
    ground_truth = ground_truth.cuda()

    lowest_loss = float("inf")

    total_elements = len(ground_truth.flatten().tolist())
    chunk_size = largest_factor_less_than((total_elements // num_channels), max_chunk_size)

    #assert chunk_size % num_gpus  == 0, "Chunk size must be a multiple of num_gpus for it to split evenly amongst gpus"
    #assert total_elements % chunk_size == 0, "Total elements must be a multiple of chunk size"

    num_chunks = (total_elements // num_channels) // chunk_size
    remaining_elements = (total_elements // num_channels) % chunk_size # remaining elements after dividing by chunk_size
    print("Using batch size", chunk_size, "with total elements", total_elements, "and num chunks", num_chunks)
    
    print("model_input shape", model_input.shape, "ground_truth shape", ground_truth.shape)

    for step in range(total_steps):
        optim.zero_grad()
        loss = 0.
        
        for chunk_num in range(num_chunks):
            start = chunk_num * chunk_size
            end = (chunk_num + 1) * chunk_size
            
            model_input_batch = model_input[:, start:end, :]
            ground_truth_batch = ground_truth[:, start:end, :]
            
            if chunk_size % (2 * in_features) == 0 and chunk_size % (2 * out_features) == 0:
                model_input_batch = model_input_batch.view(2, -1, in_features)
                ground_truth_batch = ground_truth_batch.view(2, -1, out_features)
            else:
                model_input_batch = model_input[:, start:end, :].view(1, -1, in_features)
                ground_truth_batch = ground_truth[:, start:end, :].view(1, -1, out_features)
            
            model_prediction, _ = model(model_input_batch)  
            
            assert model_prediction.shape == ground_truth_batch.shape
            miniloss = ((model_prediction - ground_truth_batch) ** 2).mean() 
            miniloss /= num_chunks
            miniloss.backward()
            
            loss += miniloss.item()
        
        if remaining_elements != 0:
            print("Hit remaining elements")
            # Process the remaining elements
            start = num_chunks * chunk_size
            end = start + remaining_elements
            
            model_input_batch = model_input[:, start:end, :]
            ground_truth_batch = ground_truth[:, start:end, :]
            
            model_input_batch = model_input_batch.view(1, -1, in_features)
            ground_truth_batch = ground_truth_batch.view(1, -1, out_features)
            
            model_prediction, _ = model(model_input_batch)  
            
            assert model_prediction.shape == ground_truth_batch.shape
            miniloss = ((model_prediction - ground_truth_batch) ** 2).mean() 
            miniloss /= num_chunks
            miniloss *= (end - start) / chunk_size
            miniloss.backward()
            
            loss += miniloss.item()
        
        loss = torch.Tensor([loss])
        lowest_loss = min(lowest_loss, float(loss.item()))
        if np.isnan(loss.item()):
            print("Got nan loss")
            exit()
        
        if step % 100 == 0:
            print("Step %d, Lowest loss %0.10f" % (step, lowest_loss), format(lowest_loss, ".4e"), "Took", round(time.time() - time_measurement, 2), "seconds")
        
        time_measurement = time.time()

        optim.step()
    
    # Clear up memory if that's the problem
    del model_input
    del ground_truth
    torch.cuda.empty_cache()
    
    return model

os.chdir(data_folder)
os.chdir(dataset)
    
for file in os.listdir()[:12]:
    if version != "Bicubic":
        trained_model = train_network(file)
    
    for srf_factor in SRF_factors:
        HR_sidelength_to_use = LR_sidelength_to_use * srf_factor

        # Initialize paths
        mse_list_path = os.path.join('serialized', 'superres_data', f'{dataset}_{version}_{(LR_sidelength_to_use, HR_sidelength_to_use)}_losses')
        psnr_list_path = os.path.join('serialized', 'superres_data', f'{dataset}_{version}_{(LR_sidelength_to_use, HR_sidelength_to_use)}_psnrs')
        corrs_list_path = os.path.join('serialized', 'superres_data', f'{dataset}_{version}_{(LR_sidelength_to_use, HR_sidelength_to_use)}_corrs')

        HR_sidelength_to_use = LR_sidelength_to_use * srf_factor
        HR_coords = get_mgrid(HR_sidelength_to_use)
        HR_prediction = forward_pass_in_chunks(trained_model, HR_coords)
        HR_gt = get_image_object_tensor(HR_sidelength_to_use, file)[0:num_channels, :, :]

        HR_gt = HR_gt.cuda()
        HR_gt = HR_gt.permute(1, 2, 0)
        out_features = HR_gt.shape[-1]
        HR_gt = HR_gt.view(-1, out_features)
        
        mse = get_MSE(HR_gt, HR_prediction)
        print("MSE with HR", mse.item())
        psnr = get_PSNR(HR_gt, HR_prediction)
        print("PSNR with HR", psnr)
        
        # Normalize the tensors to [0, 1] range if they are not already
        HR_prediction = (HR_prediction - HR_prediction.min()) / (HR_prediction.max() - HR_prediction.min())
        HR_gt = (HR_gt - HR_gt.min()) / (HR_gt.max() - HR_gt.min())
        
        HR_prediction = HR_prediction.permute(1, 0).reshape(out_features, HR_sidelength_to_use, HR_sidelength_to_use)
        HR_gt = HR_gt.permute(1, 0).reshape(out_features, HR_sidelength_to_use, HR_sidelength_to_use)
        
        # Write the amplitude correlation
        # Get two 2D arrays
        a = np.abs(np.fft.fft2(HR_prediction.detach().cpu().numpy())).flatten().tolist()
        b = np.abs(np.fft.fft2(HR_gt.detach().cpu().numpy())).flatten().tolist()

        assert len(a) == len(b)

        # Calculate the correlation between the two arrays using the cross-correlation method
        corr = correlation(a, b)
        print("Got corr of", corr)
        
        # Common operations
        mse_list = read(mse_list_path)
        mse_list.append(mse.item())
        write(mse_list_path, mse_list)
        psnr_list = read(psnr_list_path)
        psnr_list.append(psnr)
        write(psnr_list_path, psnr_list)
        corrs_list = read(corrs_list_path)
        corrs_list.append(corr)
        write(corrs_list_path, corrs_list)
