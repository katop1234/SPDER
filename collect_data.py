import subprocess, os, sys
import torch, random

assert torch.cuda.is_available()

# CHANGE THIS BEFORE RUNNING ANYTHING
media_type = sys.argv[1]
MODEL_VERSION = sys.argv[2]
# CHANGE THIS BEFORE RUNNING ANYTHING

base_dir = ...

def get_list_of_all_files(media_type):
    if media_type == "image":
        folder = base_dir + "/superres/DIV2K_train_HR/"
    elif media_type == "video":
        folder = base_dir + "/videos/UCF-101/"
    elif media_type == "superres":
        folder = base_dir + "/superres"
    elif media_type == "frameinter":
        folder = base_dir + "/frameinter"
    elif media_type == "audio" or media_type == "audio_interp":
        folder = base_dir + "/audio/ESC-50-master/audio/"

    # Get the current working directory
    pwd = os.getcwd()
    cwd = folder 

    # Get a list of all files in the current directory and its subdirectories
    files = []
    for dirpath, dirnames, filenames in os.walk(cwd):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))

    os.chdir(pwd)
    return files

assert MODEL_VERSION in ["SPDER", "SIREN", "RELU", "RELU_PE", "RELU_FFN"]

count = 0
all_files = (get_list_of_all_files(media_type))

random.seed(4)
random.shuffle(all_files)
for file in (all_files):
    object_of_the_file = file

    if media_type == "image":
        args = ['python3', "SPDER_image.py", object_of_the_file, MODEL_VERSION]
    elif media_type == "video":
        args = ['python3', "SPDER_video.py", object_of_the_file, MODEL_VERSION]
    elif media_type == "audio":
        args = ['python3', "SPDER_audio.py", object_of_the_file, MODEL_VERSION]
    elif media_type == "audio_interp":
        args = ['python3', "SPDER_audio_interpolation.py", object_of_the_file, MODEL_VERSION]
        print("Running", args)
    subprocess.run(args)
 
    count += 1
    if count % 10 == 0:
        print("Finished", count, "files")
