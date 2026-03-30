# outdoor

import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CloudRemovalDataset(Dataset):
    def __init__(self, root_dir, normalize, crop_size):
        self.root_dir = root_dir
        self.clear_dir = os.path.join(self.root_dir, 'clear')
        self.cloud_dir = os.path.join(self.root_dir, 'hazy')
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        clear_images = [img for img in os.listdir(self.clear_dir) 
                       if img.lower().endswith(valid_extensions)]
        
        cloud_images = [img for img in os.listdir(self.cloud_dir) 
                       if img.lower().endswith(valid_extensions)]
        
        clear_set = set(clear_images)
        cloud_set = set(cloud_images)
        
        common_files = clear_set & cloud_set
        
        self.image_pairs = []
        for filename in sorted(common_files):  
            clear_path = os.path.join(self.clear_dir, filename)
            cloud_path = os.path.join(self.cloud_dir, filename)
            self.image_pairs.append((clear_path, cloud_path))
        
        print(f"clear: {len(clear_images)}")
        print(f"hazy: {len(cloud_images)}")
        print(f"match: {len(self.image_pairs)}")
        

        if len(self.image_pairs) > 0:
            print("\nMatching Example(5):")
            for i in range(min(5, len(self.image_pairs))):
                filename = os.path.basename(self.image_pairs[i][0])
                print(f"  {i+1}. {filename}")
        
        transforms_list = [
            transforms.ToTensor()
        ]
        self.crop_size = crop_size
        
        if normalize:
            transforms_list.append(transforms.Normalize([0.55021476, 0.49308025, 0.47668334],
                                                        [0.24582718, 0.25504055, 0.26601505]))  
        self.transform = transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        clear_path, cloud_path = self.image_pairs[idx]
        
        clear_img = Image.open(clear_path).convert("RGB")
        cloud_img = Image.open(cloud_path).convert("RGB")
        
        if cloud_img.size != clear_img.size:
            print(f"warning: size mismatch - {os.path.basename(clear_path)}: "
                  f"clear {clear_img.size} vs cloud {cloud_img.size}")
            cloud_img = cloud_img.resize(clear_img.size, Image.Resampling.BILINEAR)
        
        if clear_img.size[0] >= self.crop_size and clear_img.size[1] >= self.crop_size:
            start_x = random.randint(0, clear_img.size[0] - self.crop_size)
            start_y = random.randint(0, clear_img.size[1] - self.crop_size)
            clear_img = clear_img.crop((start_x, start_y, start_x + self.crop_size, start_y + self.crop_size))
            cloud_img = cloud_img.crop((start_x, start_y, start_x + self.crop_size, start_y + self.crop_size))
        else:

            print(f"warning: photo {os.path.basename(clear_path)} size {clear_img.size} <  {self.crop_size}")
            clear_img = clear_img.resize((self.crop_size, self.crop_size), Image.Resampling.BILINEAR)
            cloud_img = cloud_img.resize((self.crop_size, self.crop_size), Image.Resampling.BILINEAR)
        

        if random.random() > 0.5:
            random_rotate = random.choice([90, 180, 270])
            clear_img = clear_img.rotate(random_rotate)
            cloud_img = cloud_img.rotate(random_rotate)
        
        clear_img = self.transform(clear_img)
        cloud_img = self.transform(cloud_img)
        
        sample = {'cloud_img': cloud_img, 'clear_img': clear_img}
        return sample

