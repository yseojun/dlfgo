import os, time, random, argparse
from tqdm import tqdm, trange
import imageio
import numpy as np
import cv2
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
# from torch.cuda.amp import autocast, GradScaler
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

from lib import utils,dlfgo_model
from lib.dataset import LightFieldVideoDataset
from lib.lf_video_util import get_lf_video_validation_idx


def config_parser():
    '''Define command line arguments
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--render_time_sweep", action='store_true',
                        help='render across time [0,1] with a fixed pose')
    parser.add_argument("--num_frames", type=int, default=None,
                        help='number of frames for time sweep (defaults to frame_num)')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')
    # lightfield options
    parser.add_argument("--ray_type", type=str, default='twoplane', help='lightfield type')
    parser.add_argument("--ndc", action='store_true')
    parser.add_argument("--grid_num", type=int, nargs='+', default=0, help="A list of integers to process")
    parser.add_argument("--grid_dim", type=int, default=16, help='color voxel grid dimension')
    parser.add_argument("--grid_size", type=int, default=9, help='viewpoint grid size (deprecated, use grid_size_x and grid_size_y)')
    parser.add_argument("--grid_size_x", type=int, default=None, help='viewpoint grid size in x direction')
    parser.add_argument("--grid_size_y", type=int, default=None, help='viewpoint grid size in y direction')
    parser.add_argument("--frame_num", type=int, default=40, help='number of frames')
    parser.add_argument("--start_frame", type=int, default=0, help='number of start frames')
    parser.add_argument("--mlp_depth", type=int, default=4, help='color voxel grid depth')
    parser.add_argument("--mlp_width", type=int, default=128, help='color voxel grid width')
    parser.add_argument("--lr_grid", type=float, default=1e-03, help='lr of grid')
    parser.add_argument("--wd_grid", type=float, default=0, help='lr weight decay of grid')
    parser.add_argument("--lr_mlp", type=float, default=1e-03, help='lr of the mlp')
    parser.add_argument("--batch_size", type=int, default=129600, help='batch size (number of random rays per optimization step)')
    parser.add_argument("--load2gpu_on_the_fly", action='store_true',)
    parser.add_argument("--basedir", type=str, default='/data/ysj/result/dlfgo_logs/250829', help='base directory')
    parser.add_argument("--expname", type=str, default='', help='experiment name')
    parser.add_argument("--datadir", type=str, default='/data/ysj/dataset/stanford_half', help='data directory')
    parser.add_argument("--factor", type=int, default=1, help='downsample factor for LLFF data')
    parser.add_argument("--llffhold", type=int, default=8, help='holdout for LLFF data')
    parser.add_argument("--dataset_type", type=str, default='llff', help='dataset type')
    parser.add_argument("--dataset_name", type=str, default='stanford', help='dataset name')
    parser.add_argument("--epoch", default=1000, type=int)
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--pe",type=int, default=0)
    parser.add_argument("--decomp", type=str, default='4d')
    parser.add_argument("--levels", type=int, default=1)
    # ray min/max for grid normalization
    parser.add_argument("--ray_min", type=float, nargs=5, default=[-22.5, -5.0, -1.0, -1.0, 0.0],
                        help='ray min values for normalization [u_min, v_min, s_min, t_min, time_min]')
    parser.add_argument("--ray_max", type=float, nargs=5, default=[22.5, 5.0, 1.0, 1.0, 1.0],
                        help='ray max values for normalization [u_max, v_max, s_max, t_max, time_max]')
    # logging/saving options
    parser.add_argument("--save" , action='store_true')
    # parser.add_argument("--i_weights", type=int, default=1000000,
    #                     help='frequency of weight ckpt saving')
    parser.add_argument("--save_epoch", type=int, default=20,
                        help='frequency of model saving')
    return parser

@torch.no_grad()
def warmup(model, ray_type, H, W, K, c2w, time_scalar=None, focal_depth=None):
    '''Warm up the model for faster inference.
    '''
    if ray_type == 'twoplane':
        ray = utils.get_rays_of_a_view_twoplane(H, W, K, c2w, focal_depth)
    elif ray_type == 'oneplane':
        ray = utils.get_rays_of_a_view_oneplane(H, W, K, c2w)
    # elif ray_type == 'plucker':
    #     ray = utils.get_rays_of_a_view_plucker(H, W, K, c2w)
    else:
        raise ValueError(f'Unknown ray_type: {ray_type}')
    if time_scalar is not None:
        time_tensor = torch.full((H*W, 1), float(time_scalar), dtype=ray.dtype, device=ray.device)
        ray = torch.cat([ray, time_tensor], dim=-1)
    _ = model(ray).reshape(H,W,-1)
    del _

@torch.no_grad()
def render_from_dataset(model, dataset, savedir=None, dump_images=True, expdir=None,
                        eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images from dataset; run evaluation if gt given.
    '''
    rgbs = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    render_time = []
    
    device = next(model.parameters()).device
    
    for i in tqdm(range(len(dataset))):
        render_time0 = torch.cuda.Event(enable_timing=True)
        render_time1 = torch.cuda.Event(enable_timing=True)
        
        # Get data from dataset
        data = dataset[i]
        uvst = data['uvst'].to(device)  # [H, W, 4]
        cur_time = data['cur_time'].to(device)  # scalar
        gt_img = data['img'].cpu().numpy()  # [H, W, 3]
        
        H, W, _ = uvst.shape
        
        # Expand time to match spatial dimensions
        cur_time_expanded = torch.full((H, W, 1), float(cur_time), dtype=uvst.dtype, device=device)
        
        # Concatenate uvst and time
        ray = torch.cat([uvst, cur_time_expanded], dim=-1)  # [H, W, 5]
        ray = ray.reshape(-1, 5)  # [H*W, 5]
        
        # Render
        render_time0.record()
        render_result = model(ray).reshape(H, W, -1)
        render_time1.record()
        torch.cuda.synchronize()
        render_time.append(torch.cuda.Event.elapsed_time(render_time0, render_time1))
        
        rgb = render_result.cpu().numpy()
        rgbs.append(rgb)
        
        if i == 0:
            print('Testing', rgb.shape)
        
        # Compute metrics
        p = -10. * np.log10(np.mean(np.square(rgb - gt_img)))
        psnrs.append(p)
        if eval_ssim:
            ssims.append(utils.rgb_ssim(rgb, gt_img, max_val=1))
        if eval_lpips_alex:
            lpips_alex.append(utils.rgb_lpips(rgb, gt_img, net_name='alex', device=device))
        if eval_lpips_vgg:
            lpips_vgg.append(utils.rgb_lpips(rgb, gt_img, net_name='vgg', device=device))
    
    # Print statistics
    if len(psnrs):
        avg_psnr = np.mean(psnrs)
        test_str = f'Testing psnr: {avg_psnr} (avg) \n'
        if eval_ssim: test_str += f' / Testing ssim: {np.mean(ssims)} (avg) \n'
        if eval_lpips_vgg: test_str += f' / Testing lpips (vgg): {np.mean(lpips_vgg)} (avg) \n'
        if eval_lpips_alex: test_str += f' / Testing lpips (alex): {np.mean(lpips_alex)} (avg) \n'
        if len(render_time): test_str += f' / Testing render time: {np.mean(render_time)} (avg) '
        print(test_str)
        if expdir is not None:
            with open(f'{expdir}/test_results.txt', 'a') as f:
                f.write(test_str+'\n')
    
    # Save images
    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            # save rgb image
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            # save diff image
            gt_img = dataset[i]['img'].cpu().numpy()
            diff = rgbs[i] - gt_img
            diff8 = utils.to8b(abs(diff)*10)
            heatmap = cv2.applyColorMap(diff8, cv2.COLORMAP_RAINBOW)
            filename = os.path.join(savedir, '{:03d}_diff.png'.format(i))
            imageio.imwrite(filename, heatmap)
    
    rgbs = np.array(rgbs)
    return rgbs

@torch.no_grad()
def render_viewpoints(model, ray_type, render_set, render_poses, HW, Ks,
                        gt_imgs=None, savedir=None, dump_images=True, expdir=None, render_factor=0,
                        eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False,
                        times=None, focal_depth=None):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    # assert len(render_poses) == len(HW) and len(HW) == len(Ks)
    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor
        
    rgbs = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    render_time = []
    H, W = HW[0]
    K = Ks[0]
    # focal_depth = torch.tensor([focal_depth], dtype=torch.float32)
    # model.eval()
    # warm up
    warmup(model, ray_type, H, W, K, render_poses[0],
           time_scalar=(times[0] if (times is not None and len(times) > 0) else None), focal_depth=focal_depth)
 
    if ray_type == 'twoplane' or ray_type == 'oneplane':
        # H = 1024
        # W = 1024
        # K = torch.tensor([[H, 0, H/2], [0, H, W/2], [0, 0, 1]], dtype=torch.float32)
        
        for i, c2w in enumerate(tqdm(render_poses)):
            render_time0 = torch.cuda.Event(enable_timing=True)
            render_time1 = torch.cuda.Event(enable_timing=True)
            c2w = torch.Tensor(c2w)
            if ray_type == 'twoplane':
                ray = utils.get_rays_of_a_view_twoplane(H, W, K, c2w, focal_depth)
            elif ray_type == 'oneplane':
                ray = utils.get_rays_of_a_view_oneplane(H, W, K, c2w)
            else:
                raise ValueError(f'Unknown ray_type: {ray_type}')
            # append time if provided
            if times is not None:
                t_val = float(times[i])
                time_tensor = torch.full((H*W, 1), t_val, dtype=ray.dtype, device=ray.device)
                ray = torch.cat([ray, time_tensor], dim=-1)
            render_time0.record()
            render_result = model(ray).reshape(H,W,-1)
            render_time1.record()
            torch.cuda.synchronize()
            render_time.append(torch.cuda.Event.elapsed_time(render_time0, render_time1))
            rgb = render_result.cpu().numpy()
            rgbs.append(rgb)
            if i==0:
                print('Testing', rgb.shape)
            if gt_imgs is not None and render_factor==0 and i < len(gt_imgs):
                p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
                psnrs.append(p)
                if eval_ssim:
                    ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
                if eval_lpips_alex:
                    lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
                if eval_lpips_vgg:
                    lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))
                    
    if len(psnrs):
        avg_psnr = np.mean(psnrs)
        test_str = f'Set: {render_set} / Testing psnr: {avg_psnr} (avg) \n'
        if eval_ssim: test_str += f' / Testing ssim: {np.mean(ssims)} (avg) \n'
        if eval_lpips_vgg: test_str += f' / Testing lpips (vgg): {np.mean(lpips_vgg)} (avg) \n'
        if eval_lpips_alex: test_str += f' / Testing lpips (alex): {np.mean(lpips_alex)} (avg) \n'
        if len(render_time): test_str += f' / Testing render time: {np.mean(render_time)} (avg) '
        print(test_str)
        if expdir is not None:
            with open(f'{expdir}/test_results.txt', 'a') as f:
                f.write(test_str+'\n')
                
    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            # save rgb image
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            # save diff image
            diff = rgbs[i] - gt_imgs[i]
            diff8 = utils.to8b(abs(diff)*10)
            heatmap = cv2.applyColorMap(diff8, cv2.COLORMAP_RAINBOW)
            filename = os.path.join(savedir, '{:03d}_diff.png'.format(i))
            imageio.imwrite(filename, heatmap)
    rgbs = np.array(rgbs)
    return rgbs
    

def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# load_everything function removed - no longer needed with dataset-based approach

def print_cuda_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # 메모리 사용량 (MB)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # 예약된 메모리 (MB)
    print(f"Allocated memory: {allocated:.2f} MB / Reserved memory: {reserved:.2f} MB")
    
def create_new_model(args, ray_min, ray_max):
    print(f'\033[96muse LightField {args.ray_type} grid\033[0m')
    if args.ray_type == 'twoplane' or args.ray_type == 'oneplane':        
        model = dlfgo_model.DLFGO_twoplane(
            ray_min=ray_min, ray_max=ray_max, grid_num=args.grid_num,
            ray_type='twoplane', grid_dim=args.grid_dim,
            mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
            pe=args.pe,
            decomp=args.decomp, levels=args.levels,)
    else:
        raise ValueError(f'Unknown ray_type: {args.ray_type}')
    model = model.to(device)
    params = []
    for name in dir(model):
        if "grid" in name:
            attr = getattr(model, name)
            if hasattr(attr, "parameters"):
                params.append({'params': attr.parameters(), 'lr': args.lr_grid})

    params.append({'params': model.mlp.parameters(), 'lr': args.lr_mlp})

    optimizer = torch.optim.Adam(params)
    return model, optimizer

def load_existed_model(args, reload_ckpt_path):
    if args.ray_type == 'twoplane':
        model_class = dlfgo_model.DLFGO_twoplane
    # elif args.ray_type == 'plucker':
    #     model_class = dlfgo_model.DirectLF5DGO
    else:
        raise ValueError(f'Unknown ray_type: {args.ray_type}')
    model = utils.load_model(model_class, reload_ckpt_path).to(device)
    
    params = []
    for name in dir(model):
        if "grid" in name:
            attr = getattr(model, name)
            if hasattr(attr, "parameters"):
                params.append({'params': attr.parameters(), 'lr': args.lr_grid})

    params.append({'params': model.mlp.parameters(), 'lr': args.lr_mlp})
    optimizer = torch.optim.Adam(params)
    model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    return model, optimizer, start

def scene_rep_reconstruction(args):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(args.basedir, args.expname, f'train_last.tar')
    
    if args.no_reload:
        reload_ckpt_path = None
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None
    
    # Use user-specified ray_min and ray_max from args
    ray_min = torch.tensor(args.ray_min, dtype=torch.float32)
    ray_max = torch.tensor(args.ray_max, dtype=torch.float32)
    print(f'Using ray_min: {ray_min}, ray_max: {ray_max}')
    
    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction: train from scratch')
        model, optimizer = create_new_model(args, ray_min, ray_max)
        start = 0
    else:
        print(f'scene_rep_reconstruction: reload from {reload_ckpt_path}')
        model, optimizer, start = load_existed_model(args, reload_ckpt_path)
    
    # Create dataset and dataloaders
    dataset = LightFieldVideoDataset(
        basedir=args.datadir,
        grid_size_x=args.grid_size_x,
        grid_size_y=args.grid_size_y,
        train_frames_num=args.frame_num,
        start_frames_num=args.start_frame
    )
    
    # Get validation indices and create train/val splits
    val_indices = get_lf_video_validation_idx(args.grid_size_x, args.grid_size_y, args.frame_num)
    all_indices = list(range(len(dataset)))
    train_indices = [idx for idx in all_indices if idx not in val_indices]
    
    # Create subsets and dataloaders
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=0, generator=torch.Generator(device='cuda'))
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=0)

    # Calculate epoch and iteration parameters
    # Get a sample to calculate rays per image batch
    sample_batch = next(iter(train_loader))
    B, H, W, _ = sample_batch['uvst'].shape
    rays_per_image_batch = B * H * W
    sub_batches_per_image_batch = (rays_per_image_batch + args.batch_size - 1) // args.batch_size
    iter_per_epoch = len(train_loader) * sub_batches_per_image_batch
    total_epochs = args.epoch
    total_iter = total_epochs * iter_per_epoch
    start_epoch = start // iter_per_epoch if start > 0 else 0
    print(f'Training image batches: {len(train_loader)} / Rays per image batch: {rays_per_image_batch} / Ray batch size: {args.batch_size} / Sub-batches per image batch: {sub_batches_per_image_batch}')
    print(f'Total Epochs: {total_epochs} / Total Iter: {total_iter} / Iter per epoch: {iter_per_epoch}')
    
    torch.cuda.empty_cache()
    psnr_lst = []
    psnr_val_lst = []
    time0 = time.time()
    global_step = start - 1 if start > 0 else -1
    print("before training, cuda memory usage check")
    print_cuda_memory_usage()
    
    # Training loop with DataLoader
    for epoch in range(start_epoch, total_epochs):
        for batch_idx, batch_data in enumerate(train_loader):
            # Extract batch data
            uvst = batch_data['uvst'].to(device)  # [5, H, W, 4]
            cur_time = batch_data['cur_time'].to(device)  # [5]
            img = batch_data['img'].to(device)  # [5, H, W, 3]
            
            # Expand cur_time to match spatial dimensions
            B, H, W, _ = uvst.shape
            cur_time_expanded = cur_time.view(B, 1, 1, 1).expand(B, H, W, 1)  # [5, H, W, 1]
            
            # Concatenate uvst and time
            ray = torch.cat([uvst, cur_time_expanded], dim=-1)  # [5, H, W, 5]
            
            # Flatten ray and img
            ray = ray.reshape(-1, 5)  # [5*H*W, 5]
            rgb = img.reshape(-1, 3)  # [5*H*W, 3]
            
            # Shuffle all rays
            total_rays = ray.shape[0]
            indices = torch.randperm(total_rays, device=device)
            
            # Sample and train until all rays are consumed
            for sub_batch_idx in range(0, total_rays, args.batch_size):
                global_step += 1
                
                # Get ray batch
                end_idx = min(sub_batch_idx + args.batch_size, total_rays)
                batch_indices = indices[sub_batch_idx:end_idx]
                ray_batch = ray[batch_indices]
                rgb_batch = rgb[batch_indices]
                
                # Forward pass and loss computation
                optimizer.zero_grad(set_to_none=True)
                render_result = model(ray_batch, global_step)
                loss = F.mse_loss(render_result, rgb_batch)
                psnr = utils.mse2psnr(loss.detach())
                loss = loss * 1000
                
                # Backward pass
                loss.backward()
                optimizer.step()
                psnr_lst.append(psnr.item())
                
                # Update learning rate
                decay_factor = 0.995 ** (1/iter_per_epoch)
                for i_opt_g, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = param_group['lr'] * decay_factor
                
                # Logging and validation
                if global_step%(1*iter_per_epoch)==0 or (global_step%(1*iter_per_epoch)==0 and global_step < (10*iter_per_epoch)) or (global_step%2000==0 and global_step < (10*iter_per_epoch)):
                    eps_time = time.time() - time0
                    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
                    write_str = f'expname / {args.expname} / '
                    write_str += f'Eps / {eps_time_str} / '
                    write_str += f'grid lr / {optimizer.param_groups[0]["lr"]:.8f} / '
                    write_str += f'mlp lr / {optimizer.param_groups[1]["lr"]:.8f} / '
                    write_str += f'Loss / {loss.item():.7f} / '
                    write_str += f'global_step / {global_step:6d} / '
                    write_str += f'epoch / {global_step//iter_per_epoch:4d} / Train PSNR / {np.mean(psnr_lst):5.2f} / '
                    
                    # Validation
                    if global_step%(10*iter_per_epoch)==0 or (global_step%(1*iter_per_epoch)==0 and global_step < (10*iter_per_epoch)):
                        with torch.no_grad():
                            for val_batch_data in val_loader:
                                # Extract validation batch data
                                uvst_val = val_batch_data['uvst'].to(device)
                                cur_time_val = val_batch_data['cur_time'].to(device)
                                img_val = val_batch_data['img'].to(device)
                                
                                # Process similar to training
                                B_val, H_val, W_val, _ = uvst_val.shape
                                cur_time_val_expanded = cur_time_val.view(B_val, 1, 1, 1).expand(B_val, H_val, W_val, 1)
                                ray_val = torch.cat([uvst_val, cur_time_val_expanded], dim=-1)
                                ray_val = ray_val.reshape(-1, 5)
                                rgb_val = img_val.reshape(-1, 3)
                                
                                # Compute validation loss
                                render_result_val = model(ray_val, global_step)
                                mse_loss_val = F.mse_loss(render_result_val, rgb_val)
                                psnr_val = utils.mse2psnr(mse_loss_val.detach())
                                psnr_val_lst.append(psnr_val.item())
                            
                            write_str += f'Val PSNR / {np.mean(psnr_val_lst):5.2f} / '
                    
                    tqdm.write(write_str)
                    with open(os.path.join(args.basedir, args.expname, 'training_result.txt'), 'a') as f:
                        f.write(write_str+'\n')
                    psnr_lst = []
                    psnr_val_lst = []
                
                # Save model checkpoints
                if global_step%(args.save_epoch*iter_per_epoch)==0 and global_step > 0:
                    model_path = os.path.join(args.basedir, args.expname, 'model', f'train_last_{global_step//(iter_per_epoch)}epoch.tar')
                    torch.save({
                        'global_step': global_step,
                        'model_kwargs': model.get_kwargs(),
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, model_path)
                    print(f'scene_rep_reconstruction: saved checkpoints at', model_path)
    
    if global_step != -1:
        # grid_path = os.path.join(args.basedir, args.expname, 'model', f'train_last_{global_step//(iter_per_epoch)}epoch_grid.pt')
        # torch.save(model.grid.state_dict(), grid_path)
        # print(f'scene_rep_reconstruction: saved grid at', grid_path)
        # mlp_path = os.path.join(args.basedir, args.expname, 'model', f'train_last_{global_step//(iter_per_epoch)}epoch_mlp.pt')
        # torch.save(model.mlp.state_dict(), mlp_path)
        # print(f'scene_rep_reconstruction: saved mlp at', mlp_path)
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scaler_state_dict': scaler.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction: saved checkpoints at', last_ckpt_path)
        weight_path = os.path.join(args.basedir, args.expname, f'train_last_weight.pt')
        torch.save({'model_kwargs': model.get_kwargs(),
                    'model_state_dict': model.state_dict(),}
                   , weight_path)
        print(f'scene_rep_reconstruction: saved weights at', weight_path)
    
        

def train(args):
    torch.cuda.empty_cache()
    # init
    eps_time = time.time()
    os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
    os.makedirs(os.path.join(args.basedir, args.expname, 'model'), exist_ok=True)
    with open(os.path.join(args.basedir, args.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    
    scene_rep_reconstruction(args)
    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')
    # wandb.finish()
    
if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    data_name = os.path.basename(args.datadir)
    
    # Handle grid_size_x and grid_size_y
    if args.grid_size_x is None and args.grid_size_y is None:
        # Use old grid_size for backward compatibility
        args.grid_size_x = args.grid_size
        args.grid_size_y = args.grid_size
    elif args.grid_size_x is None:
        args.grid_size_x = args.grid_size_y
    elif args.grid_size_y is None:
        args.grid_size_y = args.grid_size_x
    
    if args.grid_num == 0:
        if args.dataset_name == 'cam':
            args.grid_num = [2, 1, 960, 540, args.frame_num]
        else:
            args.grid_num = [12, 3, 240, 135, args.frame_num // 4]
        
    # default num_frames for time sweep
    if args.num_frames is None:
        args.num_frames = args.frame_num
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(args.gpuid)
        print('>>> Using GPU: {}'.format(args.gpuid))
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # torch.set_default_tensor_type('torch.cuda.HalfTensor')
    else:
        print('>>> Using CPU')
    seed_everything()
    
    # Train the model
    if not args.render_only:
        train(args)
        
    # load model for rendring
    if args.render_test or args.render_train or args.render_time_sweep:
        ckpt_path = os.path.join(args.basedir, args.expname, 'train_last.tar')
        # ckpt_path = os.path.join(args.basedir, args.expname, 'model/train_last_400epoch.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        model_class = dlfgo_model.DLFGO_twoplane
        # elif args.ray_type == 'plucker':
        #     model_class = dlfgo_model.DirectLF5DGO
        model = utils.load_model(model_class, ckpt_path).to(device)
        
        # Create dataset for rendering
        dataset = LightFieldVideoDataset(
            basedir=args.datadir,
            grid_size_x=args.grid_size_x,
            grid_size_y=args.grid_size_y,
            train_frames_num=args.frame_num,
            start_frames_num=args.start_frame
        )
        
        # Get validation indices
        val_indices = get_lf_video_validation_idx(args.grid_size_x, args.grid_size_y, args.frame_num)
        val_dataset = Subset(dataset, val_indices)
    
    # render trainset and eval
    if args.render_train:
        print('WARNING: render_train is not yet implemented for dataset-based approach')
        # TODO: Implement render_train using dataset or remove this option
    
    # render testset and eval (using validation dataset)
    if args.render_test:
        testsavedir = os.path.join(args.basedir, args.expname, f'render_test_{ckpt_name}')
        expdir = os.path.join(args.basedir, args.expname)
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        print(f'Rendering {len(val_dataset)} validation images from dataset...')
        rgbs = render_from_dataset(
            model=model, 
            dataset=val_dataset,
            savedir=testsavedir, 
            dump_images=args.dump_images, 
            expdir=expdir,
            eval_ssim=args.eval_ssim, 
            eval_lpips_alex=args.eval_lpips_alex, 
            eval_lpips_vgg=args.eval_lpips_vgg)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)

    # render a time sweep [0,1] with a fixed pose
    if args.render_time_sweep:
        print('WARNING: render_time_sweep is not yet implemented for dataset-based approach')
        # TODO: Implement render_time_sweep using dataset or remove this option

    print('Done')