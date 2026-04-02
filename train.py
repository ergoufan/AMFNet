
import os
import argparse
import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import CloudRemovalDataset
from model import AMFTrain
from ssim_prsn import ssim, psnr
from loss import ContrastLoss


class CheckpointManager:
    def __init__(self, save_dir, max_checkpoints=8):
        self.save_dir = Path(save_dir)
        self.max_checkpoints = max_checkpoints
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def auto_cleanup(self, current_step):
        checkpoint_files = self._get_all_checkpoints()
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        keep_list = self._select_checkpoints_to_keep(checkpoint_files, current_step)
        
        if not keep_list:  
            return
            
        deleted_count = 0
        for checkpoint_path in checkpoint_files:
            if checkpoint_path not in keep_list:
                try:
                    os.remove(checkpoint_path)
                    deleted_count += 1
                    print(f" delet: {os.path.basename(checkpoint_path)}")
                except Exception as e:
                    print(f" delet failed {checkpoint_path}: {e}")
        
        if deleted_count > 0:
            print(f" delet {deleted_count} Checkpoints, retain {len(keep_list)} 个")
    
    def _get_all_checkpoints(self):
        patterns = ['*.pth', '*.pt']
        checkpoint_files = []
        
        for pattern in patterns:
            checkpoint_files.extend(glob.glob(os.path.join(self.save_dir, pattern)))
        
        return checkpoint_files
    
    def _select_checkpoints_to_keep(self, checkpoint_files, current_step):
        keep_list = []
        
        important_patterns = ['best.pth', 'last.pth']
        for pattern in important_patterns:
            pattern_path = os.path.join(self.save_dir, pattern)
            if os.path.exists(pattern_path):
                keep_list.append(pattern_path)
        
        for checkpoint in checkpoint_files:
            if any(pattern in checkpoint for pattern in important_patterns):
                if checkpoint not in keep_list:
                    keep_list.append(checkpoint)
        
        step_checkpoints = [f for f in checkpoint_files if 'step_' in f]
        if step_checkpoints:
            step_checkpoints.sort(key=lambda x: self._extract_step_number(x), reverse=True)
            newest_step = step_checkpoints[0]
            if newest_step not in keep_list:
                keep_list.append(newest_step)
        
        if len(keep_list) < self.max_checkpoints:
            remaining_checkpoints = [f for f in checkpoint_files if f not in keep_list]
            remaining_checkpoints.sort(key=os.path.getmtime, reverse=True)
            for checkpoint in remaining_checkpoints:
                if len(keep_list) >= self.max_checkpoints:
                    break
                keep_list.append(checkpoint)
        
        return keep_list[:self.max_checkpoints]
    
    def _extract_step_number(self, filename):
        try:
            base_name = os.path.basename(filename)
            if 'step_' in base_name:
                step_str = base_name.split('step_')[1].split('.pth')[0]
                return int(step_str)
            return 0
        except:
            return 0


def freq_loss(res, clear, criterion):
    clear_fft = torch.fft.fft2(clear, dim=(-2, -1))
    clear_fft = torch.stack((clear_fft.real, clear_fft.imag), -1)
    pred_fft = torch.fft.fft2(res, dim=(-2, -1))
    pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
    loss = criterion(pred_fft, clear_fft)
    return loss


def cal_loss(res, clear_img, cloudy_img, criterion, contrast_criterion, contrast_weight=0.1):

    loss_l1 = criterion(res, clear_img)
    
    loss_contrast = contrast_criterion(res, clear_img, cloudy_img)
    
    loss_freq = freq_loss(res, clear_img, criterion)
    
    total_loss = loss_l1 + 0.1 * loss_contrast + 0.3 * loss_freq
    
    return total_loss, loss_l1, loss_contrast, loss_freq


def safe_save_model(model, filepath):
    try:
        temp_path = filepath + '.tmp'
        torch.save(model.state_dict(), temp_path)
        os.rename(temp_path, filepath)
        print(f"Checkpoint saved successfully: {os.path.basename(filepath)}")
        return True
    except Exception as e:
        print(f" Checkpoint save failed {filepath}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def evaluate(model, test_dataloader, criterion, contrast_criterion, device):
    if test_dataloader is None or len(test_dataloader) == 0:
        print("  Validation skipped: no test data")
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    
    model.eval()
    total_losses = []
    l1_losses = []
    contrast_losses = []
    freq_losses = []
    ssim_scores = []
    psnr_scores = []
    
    valid_batches = 0
    total_batches = len(test_dataloader)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            cloud_imgs = batch['cloud_img'].to(device)
            clear_imgs = batch['clear_img'].to(device)
            
            if torch.isnan(cloud_imgs).any() or torch.isinf(cloud_imgs).any():
                continue
            if torch.isnan(clear_imgs).any() or torch.isinf(clear_imgs).any():
                continue
            
            res = model(cloud_imgs)
            
            if torch.isnan(res).any() or torch.isinf(res).any():
                continue
            
            loss_total, loss_l1, loss_contrast, loss_freq = cal_loss(
                res, clear_imgs, cloud_imgs, 
                criterion, contrast_criterion, 
                contrast_weight=0.1
            )
            
            if torch.isnan(loss_total) or torch.isinf(loss_total):
                continue
            
            total_losses.append(loss_total.item())
            l1_losses.append(loss_l1.item())
            contrast_losses.append(loss_contrast.item())
            freq_losses.append(loss_freq.item())
            
            ssim_score = ssim(res, clear_imgs)
            psnr_score = psnr(res, clear_imgs)
            
            if not np.isnan(ssim_score) and not np.isinf(ssim_score):
                ssim_scores.append(ssim_score)
            if not np.isnan(psnr_score) and not np.isinf(psnr_score):
                psnr_scores.append(psnr_score)
            
            valid_batches += 1

    print(f"Validation: {valid_batches}/{total_batches} batches valid")
    
    if valid_batches == 0:
        print(" No valid validation batches")
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
        
    avg_total_loss = np.mean(total_losses) if total_losses else float('nan')
    avg_l1_loss = np.mean(l1_losses) if l1_losses else float('nan')
    avg_contrast_loss = np.mean(contrast_losses) if contrast_losses else float('nan')
    avg_freq_loss = np.mean(freq_losses) if freq_losses else float('nan')
    avg_ssim = np.mean(ssim_scores) if ssim_scores else float('nan')
    avg_psnr = np.mean(psnr_scores) if psnr_scores else float('nan')
    
    print(f'Test | Total Loss:{avg_total_loss:.4f} | L1:{avg_l1_loss:.4f} | freq:{avg_freq_loss:.4f} |'
          f'Contrast:{avg_contrast_loss:.4f} | SSIM:{avg_ssim:.4f} | PSNR:{avg_psnr:.2f}dB')
    
    return avg_total_loss, avg_l1_loss, avg_contrast_loss, avg_ssim, avg_psnr


def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, device, scheduler, args):
    checkpoint_manager = CheckpointManager(args.save_dir, max_checkpoints=8)
    
    contrast_criterion = ContrastLoss(ablation=False).to(device)
    
    train_total_losses = []
    train_l1_losses = []
    train_contrast_losses = []
    train_freq_losses = []
    train_ssim_scores = []
    train_psnr_scores = []
    
    test_total_losses = []
    test_l1_losses = []
    test_contrast_losses = []
    test_ssim_scores = []
    test_psnr_scores = []
    
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f'Checkpoint {args.resume} not found.')
        else:
            print(f'Resume from {args.resume}')
            model.load_state_dict(torch.load(args.resume, weights_only=True))
            if test_dataloader is not None and len(test_dataloader) > 0:
                test_loss, test_l1, test_contrast, test_ssim, test_psnr = evaluate(
                    model, test_dataloader, criterion, contrast_criterion, device
                )
                test_total_losses.append(test_loss)
                test_l1_losses.append(test_l1)
                test_contrast_losses.append(test_contrast)
                test_ssim_scores.append(test_ssim)
                test_psnr_scores.append(test_psnr)
    else:
        print('Train from scratch...')
    
    print(f"Initial learning rate: {args.lr}")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    best_loss = float("inf")
    step = 0
    
    for epoch in range(args.epochs):
        model.train()
        
        current_contrast_weight = 0.1
        
        epoch_total_losses = []
        epoch_l1_losses = []
        epoch_contrast_losses = []
        epoch_freq_losses = []
        epoch_ssim = []
        epoch_psnr = []
        
        if len(train_dataloader) == 0:
            print("No training data available!")
            break
        
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in progress_bar:
            step += 1
            cloud_imgs = batch['cloud_img'].to(device)
            clear_imgs = batch['clear_img'].to(device)
            
            res = model(cloud_imgs)
            
            loss, loss_l1, loss_contrast, loss_freq = cal_loss(
                res, clear_imgs, cloud_imgs,
                criterion, contrast_criterion,
                contrast_weight=current_contrast_weight
            )
            
            optimizer.zero_grad()
            loss.backward()
            
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            epoch_total_losses.append(loss.item())
            epoch_l1_losses.append(loss_l1.item())
            epoch_contrast_losses.append(loss_contrast.item())
            epoch_freq_losses.append(loss_freq.item())

            with torch.no_grad():
                ssim_score = ssim(res, clear_imgs)
                psnr_score = psnr(res, clear_imgs)
                epoch_ssim.append(ssim_score)
                epoch_psnr.append(psnr_score)
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'L1': f'{loss_l1.item():.4f}',
                'Cont': f'{loss_contrast.item():.4f}',
                'Freq': f'{loss_freq.item():.4f}',
                'W': f'{current_contrast_weight:.3f}'
            })

            if args.save_cycle > 0 and step % max(1, int(args.save_cycle * len(train_dataloader))) == 0:
                checkpoint_path = os.path.join(args.save_dir, f'step_{step}.pth')
                if safe_save_model(model, checkpoint_path):
                    checkpoint_manager.auto_cleanup(step)
        
        if len(epoch_total_losses) == 0:
            print(f" Epoch {epoch+1}: No training data processed!")
            continue
        
        avg_epoch_total_loss = np.mean(epoch_total_losses)
        avg_epoch_l1_loss = np.mean(epoch_l1_losses)
        avg_epoch_contrast_loss = np.mean(epoch_contrast_losses)
        avg_epoch_freq_loss = np.mean(epoch_freq_losses)
        
        epoch_ssim_clean = []
        for s in epoch_ssim:
            if isinstance(s, torch.Tensor):
                epoch_ssim_clean.append(s.item())
            elif not (np.isnan(s) or np.isinf(s)):
                epoch_ssim_clean.append(s)
        avg_epoch_ssim = np.mean(epoch_ssim_clean) if epoch_ssim_clean else 0.0
        
        epoch_psnr_clean = []
        for p in epoch_psnr:
            if isinstance(p, torch.Tensor):
                epoch_psnr_clean.append(p.item())
            elif not (np.isnan(p) or np.isinf(p)):
                epoch_psnr_clean.append(p)
        avg_epoch_psnr = np.mean(epoch_psnr_clean) if epoch_psnr_clean else 0.0
        
        train_total_losses.append(avg_epoch_total_loss)
        train_l1_losses.append(avg_epoch_l1_loss)
        train_contrast_losses.append(avg_epoch_contrast_loss)
        train_freq_losses.append(avg_epoch_freq_loss)
        train_ssim_scores.append(avg_epoch_ssim)
        train_psnr_scores.append(avg_epoch_psnr)
        
        print(f'\nEpoch:{epoch+1}/{args.epochs} | '
              f'Total Loss:{avg_epoch_total_loss:.4f} | '
              f'L1 Loss:{avg_epoch_l1_loss:.4f} | '
              f'Contrast Loss:{avg_epoch_contrast_loss:.4f} | '
              f'Freq Loss:{avg_epoch_freq_loss:.4f} | '
              f'Contrast Weight:{current_contrast_weight:.3f} | '
              f'SSIM:{avg_epoch_ssim:.4f} | '
              f'PSNR:{avg_epoch_psnr:.2f}dB')
        
        if test_dataloader is not None and len(test_dataloader) > 0:
            test_total_loss, test_l1_loss, test_contrast_loss, test_ssim, test_psnr = evaluate(
                model, test_dataloader, criterion, contrast_criterion, device
            )
            
            if not np.isnan(test_total_loss):
                test_total_losses.append(test_total_loss)
                test_l1_losses.append(test_l1_loss)
                test_contrast_losses.append(test_contrast_loss)
                test_ssim_scores.append(test_ssim)
                test_psnr_scores.append(test_psnr)
                
                if test_total_loss <= best_loss:
                    if safe_save_model(model, os.path.join(args.save_dir, 'best.pth')):
                        best_loss = test_total_loss
                        print(f' Saving best model (Loss: {best_loss:.4f})...')
            else:
                test_total_losses.append(float('nan'))
                test_l1_losses.append(float('nan'))
                test_contrast_losses.append(float('nan'))
                test_ssim_scores.append(float('nan'))
                test_psnr_scores.append(float('nan'))
        else:
            print(" No validation set, skipping evaluation")
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(args.save_dir, f'epoch_{epoch+1}.pth')
                if safe_save_model(model, checkpoint_path):
                    print(f' Saving model at epoch {epoch+1}')
    
    print('\n Training Complete!\n')
    safe_save_model(model, os.path.join(args.save_dir, 'last.pth'))
    print('Saved final model')
    

    plot_training_curves(
        train_total_losses, train_l1_losses, train_contrast_losses,
        train_ssim_scores, train_psnr_scores,
        test_total_losses, test_l1_losses, test_contrast_losses,
        test_ssim_scores, test_psnr_scores,
        args.curve_dir
    )


def plot_training_curves(train_total_losses, train_l1_losses, train_contrast_losses,
                         train_ssim_scores, train_psnr_scores,
                         test_total_losses, test_l1_losses, test_contrast_losses,
                         test_ssim_scores, test_psnr_scores,
                         curve_dir):
    if len(train_total_losses) == 0 or all(np.isnan(l) for l in train_total_losses):
        print(" No valid training data to plot curves")
        return
    
    plt.figure(figsize=(18, 10))
    
    # Loss
    plt.subplot(2, 3, 1)
    plt.plot(train_total_losses, label="Train Total Loss", linewidth=2, color='blue')
    plt.plot(train_l1_losses, label="Train L1 Loss", linestyle='--', alpha=0.7, color='green')
    plt.plot(train_contrast_losses, label="Train Contrast Loss", linestyle=':', alpha=0.7, color='red')
    if test_total_losses and not all(np.isnan(l) for l in test_total_losses):
        plt.plot(test_total_losses, label="Val Total Loss", linewidth=2, color='orange')
        plt.plot(test_l1_losses, label="Val L1 Loss", linestyle='--', alpha=0.5, color='lightgreen')
        plt.plot(test_contrast_losses, label="Val Contrast Loss", linestyle=':', alpha=0.5, color='pink')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SSIM
    plt.subplot(2, 3, 2)
    plt.plot(train_ssim_scores, label="Train SSIM", linewidth=2, color='blue')
    if test_ssim_scores and not all(np.isnan(s) for s in test_ssim_scores):
        plt.plot(test_ssim_scores, label="Val SSIM", linewidth=2, color='orange')
    plt.title('SSIM Curve')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # PSNR
    plt.subplot(2, 3, 3)
    plt.plot(train_psnr_scores, label="Train PSNR", linewidth=2, color='green')
    if test_psnr_scores and not all(np.isnan(p) for p in test_psnr_scores):
        plt.plot(test_psnr_scores, label="Val PSNR", linewidth=2, color='red')
    plt.title('PSNR Curve')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # L1 Loss
    plt.subplot(2, 3, 4)
    plt.plot(train_l1_losses, label="Train L1", linewidth=2, color='blue')
    if test_l1_losses and not all(np.isnan(l) for l in test_l1_losses):
        plt.plot(test_l1_losses, label="Val L1", linewidth=2, color='orange')
    plt.title('L1 Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Contrast Loss
    plt.subplot(2, 3, 5)
    plt.plot(train_contrast_losses, label="Train Contrast", linewidth=2, color='red')
    if test_contrast_losses and not all(np.isnan(l) for l in test_contrast_losses):
        plt.plot(test_contrast_losses, label="Val Contrast", linewidth=2, color='pink')
    plt.title('Contrast Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Contrast Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # SSIM-PSNR
    plt.subplot(2, 3, 6)
    ax1 = plt.gca()
    ax1.plot(train_ssim_scores, label="SSIM", color='blue', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('SSIM', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim([0, 1])
    
    ax2 = ax1.twinx()
    ax2.plot(train_psnr_scores, label="PSNR", color='green', linestyle='--', linewidth=2)
    ax2.set_ylabel('PSNR (dB)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    plt.title('SSIM & PSNR Trend')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    

    if not os.path.exists(curve_dir):
        os.makedirs(curve_dir)
    curve_path = os.path.join(curve_dir, 'training_curves_with_contrast.png')
    plt.savefig(curve_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    print(f'Training curves saved to {curve_path}')
    plt.show()


def main(args):
    print("Loading dataset...")
    total_dataset = CloudRemovalDataset(
        os.path.join(args.data_dir), 
        args.nor, 
        crop_size=256
    )
    
    total_size = len(total_dataset)
    print(f" Total dataset size: {total_size}")
    
    if total_size == 0:
        print(" ERROR: Dataset is empty!")
        return
    

    if total_size < 10:
        print("  Small dataset detected, adjusting parameters...")
        args.batch_size = 1
        args.epochs = min(args.epochs, 100)
        print(f"   Adjusted batch_size: {args.batch_size}, epochs: {args.epochs}")
    
 
    generator = torch.Generator().manual_seed(22)
    min_train_size = max(1, args.batch_size)
    min_test_size = 1
    
    if total_size < min_train_size + min_test_size:
        train_size = total_size
        test_size = 0
        train_set = total_dataset
        test_set = None
        print(" Using all data for training (no validation set)")
    else:
        train_size = int(total_size * 0.8)
        test_size = total_size - train_size
        
        if train_size < min_train_size:
            train_size = min_train_size
            test_size = total_size - train_size
        elif test_size < min_test_size:
            train_size = total_size - min_test_size
            test_size = min_test_size
            
        train_set, test_set = random_split(
            total_dataset, [train_size, test_size], generator=generator
        )
        print(f"Dataset split - Train: {train_size}, Test: {test_size}")
    
    print(" Creating data loaders...")
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=min(4, os.cpu_count()//2), 
        pin_memory=True, 
        drop_last=False
    )
    
    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(
            test_set, 
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=min(2, os.cpu_count()//4), 
            pin_memory=True, 
            drop_last=False
        )
    
    print(f"Train loader: {len(train_loader)} batches")
    if test_loader:
        print(f" Test loader: {len(test_loader)} batches")
    else:
        print(" Test loader: None")
    
    if len(train_loader) == 0:
        print(" ERROR: Cannot create training batches!")
        print("Trying with batch_size=1...")
        train_loader = DataLoader(
            train_set, 
            batch_size=1,
            shuffle=True, 
            num_workers=0, 
            pin_memory=True, 
            drop_last=False
        )
        print(f" Train loader (batch_size=1): {len(train_loader)} batches")
        
        if len(train_loader) == 0:
            print("CRITICAL: Still cannot create training batches. Dataset might be corrupted.")
            return
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f'  Using device: {device}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name()}')
        print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    print(" Initializing model...")
    model = nn.DataParallel(AMFTrain()).to(device)
    
    criterion = nn.L1Loss().to(device)
    
    optimizer = torch.optim.AdamW(
        params=filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr, 
        betas=(0.9, 0.999), 
        eps=1e-08
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * len(train_loader), 
        eta_min=args.min_lr
    )
    
    optimizer.zero_grad()
    
    print(" Starting training...")
    train_model(model, train_loader, test_loader, optimizer, criterion, device, scheduler, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DehazeXL model with contrast loss for cloud removal.')
    
    parser.add_argument('--data_dir', type=str, default=r'../data/OTS/train',
                        help='Path to dataset directory')
    parser.add_argument('--curve_dir', type=str, default=r'../curves/AMF',
                        help='Path to save training curves')
    parser.add_argument('--save_dir', type=str, default=r'../checkpoints/train_AMF',
                        help='Path to save checkpoints')
    parser.add_argument('--save_cycle', type=float, default=0.001,
                        help='Cycle of saving checkpoint (fraction of total steps)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training')
    
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of training epochs')
    parser.add_argument('--nor', action='store_true', default=False,
                        help='Normalize the image')
    
    parser.add_argument('--contrast_weight', type=float, default=0.1,
                        help='Fixed contrast loss weight (always 0.1)')
    
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='Gradient clipping norm (0 to disable)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(" Training Configuration with Contrast Loss")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Save directory: {args.save_dir}")
    print(f"Learning rate: {args.lr} (min: {args.min_lr})")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Cuda enabled: {not args.no_cuda and torch.cuda.is_available()}")
    print(f"Contrast weight: {args.contrast_weight} (fixed)")
    print(f"Frequency loss: always used")
    print(f"Save cycle: {args.save_cycle}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    print("=" * 60)
    
    main(args)