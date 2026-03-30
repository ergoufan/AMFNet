import os
import argparse
import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from utils import AverageMeter, pad_img, val_psnr, val_ssim
from data import ValDataset
from model import Backbone


def eval(val_loader, network, save_dir=None):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()
    network.eval()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f'Created save directory: {save_dir}')

    for batch in tqdm(val_loader, desc='evaluation'):
        hazy_img = batch['hazy'].cuda()
        clear_img = batch['clear'].cuda()

        with torch.no_grad():
            H, W = hazy_img.shape[2:]
            hazy_img = pad_img(hazy_img, 4)
            output = network(hazy_img)
            output = output.clamp(0, 1)
            output = output[:, :, :H, :W]
            
            if save_dir:
                filename = batch['filename'][0]
                save_path = os.path.join(save_dir, filename)
                save_image(output, save_path)

        psnr_tmp = val_psnr(output, clear_img)
        ssim_tmp = val_ssim(output, clear_img).item()
        PSNR.update(psnr_tmp)
        SSIM.update(ssim_tmp)

    return PSNR.avg, SSIM.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--checkpoint', type=str, default='../pre_model/haze4k.pth',
                       help='path to checkpoint')
    parser.add_argument('--hazy_dir', type=str, default='../data/Haze4K/test/hazy',
                       help='path to hazy images')
    parser.add_argument('--clear_dir', type=str, default='../data/Haze4K/test/clear',
                       help='path to clear images')
    parser.add_argument('--save_dir', type=str, default='../FM_RESULT/test_haze4k',
                       help='directory to save inference results')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='number of workers for data loading')
    parser.add_argument('--save_results', type=lambda x: (str(x).lower() == 'true'), default=True,
                       help='whether to save inference results')
    args = parser.parse_args()

    network = DataParallel(Backbone()).cuda()

    print(f"hazy_dir: {args.hazy_dir}")
    print(f"clear_dir: {args.clear_dir}")
    
    val_dataset = ValDataset(args.hazy_dir, args.clear_dir)
    val_loader = DataLoader(val_dataset,
                           batch_size=args.batch_size,
                           shuffle=False,
                           num_workers=args.num_workers,
                           pin_memory=False)

    print(f'\nLoading checkpoint from {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if not any(x in k for x in ['fe_level_2', 'fe_level_3']):
            filtered_state_dict[k] = v
    
    if all(k.startswith('module.') for k in filtered_state_dict.keys()):
        network.load_state_dict(filtered_state_dict)
    else:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in filtered_state_dict.items():
            if not k.startswith('module.'):
                k = 'module.' + k
            new_state_dict[k] = v
        network.load_state_dict(new_state_dict)
    
    print(f'Checkpoint loaded successfully')
    print(f'Number of validation images: {len(val_dataset)}')

    save_dir = args.save_dir if args.save_results else None
    if save_dir:
        print(f'Results will be saved to: {save_dir}')
    
    avg_psnr, avg_ssim = eval(val_loader, network, save_dir)
    
    print(f'\n{"="*50}')
    print(f'Evaluation Results:')
    print(f'Dataset: {args.hazy_dir}')
    print(f'PSNR: {avg_psnr:.4f}')
    print(f'SSIM: {avg_ssim:.4f}')
    if save_dir:
        print(f'Results saved to: {save_dir}')
    print(f'='*50)