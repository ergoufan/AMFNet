
import torch
from einops.layers.torch import Rearrange

def convert_cdc(w, device='cpu'):
    """Central Difference Convolution)"""
    conv_weight = w
    conv_shape = conv_weight.shape
    conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
    conv_weight_cd = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3, device=device)
    conv_weight_cd[:, :, :] = conv_weight[:, :, :]
    conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
    conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_cd)
    return conv_weight_cd

def convert_hdc(w, device='cpu'):
    """Horizontal Difference Convolution)"""
    conv_weight = w
    conv_shape = conv_weight.shape
    conv_weight_hd = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3, device=device)
    conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
    conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
    conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_hd)
    return conv_weight_hd

def convert_vdc(w, device='cpu'):
    """Vertical Difference Convolution)"""
    conv_weight = w
    conv_shape = conv_weight.shape
    conv_weight_vd = torch.zeros(conv_shape[0], conv_shape[1], 3 * 3, device=device)
    conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
    conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
    conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_vd)
    return conv_weight_vd

def convert_adc(w, device='cpu'):
    """ Average Difference Convolution)"""
    conv_weight = w
    conv_shape = conv_weight.shape
    conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
    conv_weight_ad = conv_weight - conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
    conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_ad)
    return conv_weight_ad


if __name__ == '__main__':
    saved_model_path = '../checkpoints/AMF/best.pth'
    
    print(f"LOADING: {saved_model_path}")
    
    state_dict = torch.load(saved_model_path, map_location='cpu')
    
    print(f"\n=== Checkpoint information ===")
    print(f"state_dict have {len(state_dict)} keys")
    
    print("\n=== start ===")
    simplified_ckp = {}
    
    block_count = 0
    
    base_keys = set()
    for key in state_dict.keys():
        if 'conv1_1.conv.weight' in key:
            base_key = key.replace('conv1_1.conv.weight', '')
            base_keys.add(base_key)
    
    print(f"find {len(base_keys)} blocks")
    

    for key, value in state_dict.items():

        if any(x in key for x in ['conv1_1', 'conv1_2', 'conv1_3', 'conv1_4', 'conv1_5']):
            continue
        simplified_ckp[key] = value
    

    for base_key in sorted(base_keys):
        print(f"\ndoing: {base_key}")
        

        required_keys = [
            base_key + 'conv1_1.conv.weight',
            base_key + 'conv1_1.conv.bias',
            base_key + 'conv1_2.conv.weight',
            base_key + 'conv1_2.conv.bias',
            base_key + 'conv1_3.conv.weight',
            base_key + 'conv1_3.conv.bias',
            base_key + 'conv1_4.conv.weight',
            base_key + 'conv1_4.conv.bias',
            base_key + 'conv1_5.weight',
            base_key + 'conv1_5.bias'
        ]
        
        missing = [k for k in required_keys if k not in state_dict]
        if missing:
            print(f"lack: {missing}")
            continue
     
        w_cdc = state_dict[base_key + 'conv1_1.conv.weight']
        b_cdc = state_dict[base_key + 'conv1_1.conv.bias']
        
        w_hdc = state_dict[base_key + 'conv1_2.conv.weight']
        b_hdc = state_dict[base_key + 'conv1_2.conv.bias']
        
        w_vdc = state_dict[base_key + 'conv1_3.conv.weight']
        b_vdc = state_dict[base_key + 'conv1_3.conv.bias']
        
        w_adc = state_dict[base_key + 'conv1_4.conv.weight']
        b_adc = state_dict[base_key + 'conv1_4.conv.bias']
        
        w_vc = state_dict[base_key + 'conv1_5.weight']
        b_vc = state_dict[base_key + 'conv1_5.bias']
        
        print(f"  find 5 branch: CDC, HDC, VDC, ADC, VC")
        
        w_cdc_t = convert_cdc(w_cdc, device='cpu')
        w_hdc_t = convert_hdc(w_hdc, device='cpu')
        w_vdc_t = convert_vdc(w_vdc, device='cpu')
        w_adc_t = convert_adc(w_adc, device='cpu')
        
        w_merged = w_cdc_t + w_hdc_t + w_vdc_t + w_adc_t + w_vc
        b_merged = b_cdc + b_hdc + b_vdc + b_adc + b_vc
        
        merged_weight_key = base_key + 'weight' 
        merged_bias_key = base_key + 'bias'      
        
        if not base_key.endswith('conv1.'):
            merged_weight_key = base_key + 'conv1.weight'
            merged_bias_key = base_key + 'conv1.bias'
        
        simplified_ckp[merged_weight_key] = w_merged
        simplified_ckp[merged_bias_key] = b_merged
        
        print(f" mergy: {merged_weight_key}")
        print(f" shape: {w_merged.shape}")
        
        block_count += 1
    
    print(f"\n======")
    print(f"Number of blocks processed: {block_count}")
    print(f"The simplified model includes {len(simplified_ckp)} keys")
    
    output_path = saved_model_path.replace('.pth', '_simplified.pth')
    torch.save(simplified_ckp, output_path)
    
    print(f"\n=== complete ===")
    print(f"before: {saved_model_path}")
    print(f"after: {output_path}")
    
    print(f"\n=== test ===")
    test_keys = [k for k in simplified_ckp.keys() if 'down_level1_block1.conv1.weight' in k]
    if test_keys:
        print(f"down_level1_block1.conv1.weight: {test_keys[0]}, shape: {simplified_ckp[test_keys[0]].shape}")
    
    test_keys = [k for k in simplified_ckp.keys() if 'up_level1_block1.conv1.weight' in k]
    if test_keys:
        print(f"up_level1_block1.conv1.weight: {test_keys[0]}, shape: {simplified_ckp[test_keys[0]].shape}")
    
    test_keys = [k for k in simplified_ckp.keys() if 'level3_block1.conv1.weight' in k]
    if test_keys:
        print(f"level3_block1.conv1.weight: {test_keys[0]}, shape: {simplified_ckp[test_keys[0]].shape}")