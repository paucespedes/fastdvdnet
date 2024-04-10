import os
import random
import numpy as np
import torch
from PIL import Image

class ImagesDataLoader:
    def __init__(self, batch_size, sequence_length, clean_files, noisy_files, denoised_files, crop_size):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.clean_files = clean_files
        self.noisy_files = noisy_files
        self.denoised_files = denoised_files
        self.crop_size = crop_size
        self.video_names = [d for d in os.listdir(clean_files) if
                            os.path.isdir(os.path.join(clean_files, d)) and
                            os.path.isdir(os.path.join(noisy_files, d)) and
                            os.path.isdir(os.path.join(denoised_files, d))]
        self.noise_levels = self.create_noises_dictionary(noisy_files)
        print(self.noise_levels)

    def __iter__(self):
        return self

    def __next__(self):
        # Randomly select batch_size folders from either clean_files or noisy_files
        batch_randomly_selected_videos = random.sample(self.video_names, self.batch_size)

        # Initialize batches as PyTorch tensors
        clean_batch = torch.zeros((self.batch_size, self.sequence_length, 3, self.crop_size, self.crop_size), device='cuda', dtype=torch.float)
        noisy_batch = torch.zeros((self.batch_size, self.sequence_length, 3, self.crop_size, self.crop_size), device='cuda', dtype=torch.float)
        denoised_batch = torch.zeros((self.batch_size, self.sequence_length, 3, self.crop_size, self.crop_size), device='cuda', dtype=torch.float)
        noise_levels = torch.zeros((self.batch_size, 1, 1, 1), device='cuda', dtype=torch.float)

        for i, video_name in enumerate(batch_randomly_selected_videos):
            noise_levels[i] = self.noise_levels[video_name]
            clean_batch[i], noisy_batch[i], denoised_batch[i] = self.get_frames(video_name)

        return clean_batch, noisy_batch, denoised_batch, noise_levels

    def process_image(self, image_path, x, y):
        # Load image
        image = Image.open(image_path)
        image = image.convert('RGB')  # Convert to RGB if not already

        image = image.crop((x, y, x + self.crop_size, y + self.crop_size))

        # Convert to numpy array and transpose to [C, H, W]
        image_np = np.array(image).transpose(2, 0, 1)

        # Convert the numpy array to PyTorch tensor
        image_tensor = torch.from_numpy(image_np).to(device='cuda').float()
        return image_tensor

    def get_frames(self, video_name):
        clean_path = os.path.join(self.clean_files, video_name)
        noisy_path = os.path.join(self.noisy_files, video_name)
        denoised_path = os.path.join(self.denoised_files, video_name)

        # Get all PNG image files in the folder
        clean_frame_paths = [os.path.join(clean_path, f) for f in os.listdir(clean_path) if f.endswith('.png')]
        clean_frame_paths.sort()
        noisy_frame_paths = [os.path.join(noisy_path, f) for f in os.listdir(noisy_path) if f.endswith('.png')]
        noisy_frame_paths.sort()
        denoised_frame_paths = [os.path.join(denoised_path, f) for f in os.listdir(denoised_path) if f.endswith('.png')]
        denoised_frame_paths.sort()

        ctrl_frame_idx = random.randint(self.sequence_length // 2,
                                        len(clean_frame_paths) - (1 + self.sequence_length // 2))

        ctrl_frame_image = Image.open(clean_frame_paths[ctrl_frame_idx])
        width, height = ctrl_frame_image.size
        x = random.randint(0, width - self.crop_size)
        y = random.randint(0, height - self.crop_size)

        clean_frames = torch.zeros((self.sequence_length, 3, self.crop_size, self.crop_size), device='cuda', dtype=torch.float)
        noisy_frames = torch.zeros((self.sequence_length, 3, self.crop_size, self.crop_size), device='cuda', dtype=torch.float)
        denoised_frames = torch.zeros((self.sequence_length, 3, self.crop_size, self.crop_size), device='cuda', dtype=torch.float)

        first_frame_idx = ctrl_frame_idx - self.sequence_length // 2
        last_frame_idx = ctrl_frame_idx + self.sequence_length // 2
        if self.sequence_length % 2 == 1:
            last_frame_idx += 1

        for i in range(first_frame_idx, last_frame_idx):
            clean_frames[i - first_frame_idx] = self.process_image(clean_frame_paths[i], x, y)
            noisy_frames[i - first_frame_idx] = self.process_image(noisy_frame_paths[i], x, y)
            denoised_frames[i - first_frame_idx] = self.process_image(denoised_frame_paths[i], x, y)

        return clean_frames, noisy_frames, denoised_frames

    def create_noises_dictionary(self, path):
        result_dict = {}
        # Iterate over all items in the directory
        for item in os.listdir(path):
            # Check if the item is a directory
            if os.path.isdir(os.path.join(path, item)):
                # Form the path to the subfolder's text file
                txt_file_path = os.path.join(path, item, "noise_level.txt")
                # Check if the text file exists
                if os.path.isfile(txt_file_path):
                    # Read the integer from the text file and store it in the dictionary
                    with open(txt_file_path, 'r') as file:
                        content = file.read()
                        try:
                            value = int(content.strip())/255.
                        except ValueError:
                            print(f"Warning: Unable to convert content of {txt_file_path} to an integer.")
                            continue
                        result_dict[item] = value
                else:
                    print(f"Warning: {item}.txt not found in {os.path.join(path, item)}")
        return result_dict
