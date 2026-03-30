import os, random
import torch.utils.data as data
from PIL import Image
from torchvision.transforms.functional import hflip, rotate, crop
from torchvision.transforms import ToTensor, RandomCrop, Resize


class TrainDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(TrainDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        
        self.hazy_image_list = sorted(os.listdir(hazy_path))

        self.clear_image_list = sorted(os.listdir(clear_path))
        
        print(f"雾图数量: {len(self.hazy_image_list)}")
        print(f"清晰图数量: {len(self.clear_image_list)}")
        
        self._validate_pairing()
    
    def _validate_pairing(self):

        sample_hazy = self.hazy_image_list[0]
        if '_' not in sample_hazy:
            print(f"警告: 雾图文件名 '{sample_hazy}' 不包含 '_'，可能不是ITS格式")
    
    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        
        base_name = hazy_image_name.split('_')[0]
        clear_image_name = f"{base_name}.png"
        
        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)
        
        if not os.path.exists(clear_image_path):
            clear_image_name = self.clear_image_list[index % len(self.clear_image_list)]
            clear_image_path = os.path.join(self.clear_path, clear_image_name)
            print(f"警告: {base_name}.png 不存在，使用 {clear_image_name}")
        
        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')
        
        crop_params = RandomCrop.get_params(hazy, [256, 256])
        rotate_params = random.randint(0, 3) * 90
        
        hazy = crop(hazy, *crop_params)
        clear = crop(clear, *crop_params)
        
        hazy = rotate(hazy, rotate_params)
        clear = rotate(clear, rotate_params)
        
        to_tensor = ToTensor()
        hazy = to_tensor(hazy)
        clear = to_tensor(clear)
        
        return hazy, clear
    
    def __len__(self):
        return len(self.hazy_image_list)
    

class TestDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(TestDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        
        self.hazy_image_list = sorted(os.listdir(hazy_path))
        self.clear_image_list = sorted(os.listdir(clear_path))
        
        print(f"测试集 - 雾图: {len(self.hazy_image_list)}, 清晰图: {len(self.clear_image_list)}")
    
    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        
        base_name = hazy_image_name.split('_')[0]
        clear_image_name = f"{base_name}.png"
        
        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)
        
        if not os.path.exists(clear_image_path):
            clear_image_name = self.clear_image_list[0]
            clear_image_path = os.path.join(self.clear_path, clear_image_name)
        
        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')
        
        to_tensor = ToTensor()
        hazy = to_tensor(hazy)
        clear = to_tensor(clear)
        
        return hazy, clear, hazy_image_name
    
    def __len__(self):
        return len(self.hazy_image_list)


# class ValDataset(data.Dataset):
#     def __init__(self, hazy_path, clear_path):
#         super(ValDataset, self).__init__()
#         self.hazy_path = hazy_path
#         self.clear_path = clear_path
#         self.hazy_image_list = os.listdir(hazy_path)
#         self.clear_image_list = os.listdir(clear_path)
#         self.hazy_image_list.sort()
#         self.clear_image_list.sort()

#     def __getitem__(self, index):
#         hazy_image_name = self.hazy_image_list[index]
#         clear_image_name = self.clear_image_list[index]

#         hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
#         clear_image_path = os.path.join(self.clear_path, clear_image_name)

#         hazy = Image.open(hazy_image_path).convert('RGB')
#         clear = Image.open(clear_image_path).convert('RGB')

#         to_tensor = ToTensor()

#         hazy = to_tensor(hazy)
#         clear = to_tensor(clear)

#         return {'hazy': hazy, 'clear': clear, 'filename': hazy_image_name}

#     def __len__(self):
#         return len(self.hazy_image_list)


class ValDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(ValDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        
        self.hazy_image_list = [f for f in os.listdir(hazy_path) 
                               if f.endswith('.png') or f.endswith('.jpg')]
        self.hazy_image_list.sort()
        
        self.clear_image_list = [f for f in os.listdir(clear_path) 
                                if f.endswith('.png') or f.endswith('.jpg')]
        self.clear_set = set(self.clear_image_list)  
        
        self.hazy_to_clear = {}
        self.invalid_pairs = []
        
        for hazy_name in self.hazy_image_list:
            base_id = hazy_name.split('_')[0]
            
            clear_name = f"{base_id}.png"
            
            if clear_name in self.clear_set:
                self.hazy_to_clear[hazy_name] = clear_name
            else:
                clear_name_jpg = f"{base_id}.jpg"
                if clear_name_jpg in self.clear_set:
                    self.hazy_to_clear[hazy_name] = clear_name_jpg
                else:
                    self.invalid_pairs.append(hazy_name)
                    print(f"waring:  {hazy_name} The corresponding clear image is not find")
        
        self.hazy_image_list = list(self.hazy_to_clear.keys())
        self.hazy_image_list.sort()
        
        print(f"Validation set loaded:")
        print(f"  - all the hazy photos: {len(self.hazy_image_list) + len(self.invalid_pairs)}")
        print(f"  - Valid image pair: {len(self.hazy_image_list)}")
        print(f"  - Invalid image pair: {len(self.invalid_pairs)}")
        
        if len(self.hazy_image_list) > 0:
            print(f"\n example pairing:")
            example_hazy = self.hazy_image_list[0]
            print(f"  hazy: {example_hazy}")
            print(f"  clear: {self.hazy_to_clear[example_hazy]}")

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = self.hazy_to_clear[hazy_image_name]

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        if hazy.size != clear.size:
            print(f"warning: Inconsistent size  - {hazy_image_name}")
            print(f"  hazy size: {hazy.size}")
            print(f"  clear size: {clear.size}")
            clear = clear.resize(hazy.size, Image.Resampling.BILINEAR)

        to_tensor = ToTensor()
        hazy_tensor = to_tensor(hazy)
        clear_tensor = to_tensor(clear)

        return {
            'hazy': hazy_tensor, 
            'clear': clear_tensor, 
            'filename': hazy_image_name,
            'clear_filename': clear_image_name  
        }

    def __len__(self):
        return len(self.hazy_image_list)