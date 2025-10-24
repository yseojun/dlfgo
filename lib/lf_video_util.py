import os


def get_lf_video_validation_idx(grid_size_x, grid_size_y, frame_num):
    """
    Get validation indices for light field video data.
    
    Args:
        grid_size_x: Grid size in x direction
        grid_size_y: Grid size in y direction  
        frame_num: Number of frames
        
    Returns:
        List of validation indices
    """
    result = []
    mid_y = grid_size_y // 2
    mid_x = grid_size_x // 2
    
    # Determine validation indices for y
    if mid_y <= 2:
        val_idx_y = [mid_y]
    else:
        min_y = mid_y // 2
        max_y = mid_y + min_y
        val_idx_y = [min_y, mid_y, max_y]
    
    # Determine validation indices for x
    if mid_x <= 2:
        val_idx_x = [mid_x]
    else:
        min_x = mid_x // 2
        max_x = mid_x + (mid_x - min_x)
        val_idx_x = [min_x, mid_x, max_x]
    
    for y in val_idx_y:
        for x in val_idx_x:
            for f in range(frame_num):
                idx = (y * grid_size_x + x) * frame_num + f
                result.append(idx)
    print("validation index : ", result)
    return result

def get_frame_from_idx(idx, grid_size_x, grid_size_y, frame_num):
    """
    Get frame coordinates from index.
    
    Args:
        idx: Index
        grid_size_x: Grid size in x direction
        grid_size_y: Grid size in y direction
        frame_num: Number of frames
        
    Returns:
        Tuple of (y, x, f) coordinates
    """
    y = idx // (grid_size_x * frame_num)
    x = (idx % (grid_size_x * frame_num)) // frame_num
    f = idx % frame_num
    return y, x, f

def get_view_dirs(imgdir):
    return sorted([os.path.join(imgdir, d) for d in os.listdir(imgdir) if os.path.isdir(os.path.join(imgdir, d))])

def get_imgs_from_view_dirs(view_dirs, frame_num):
    imgs = []
    for view_dir in view_dirs:
        imgs.extend(sorted([os.path.join(view_dir, f) for f in os.listdir(os.path.join(view_dir)) if f.endswith('.png')])[:frame_num])
    return imgs

def __main__():
    # print(get_lf_video_validation_idx(17, 2))
    # print(get_frame_from_idx(410, 9, 20))
    imgdir = '/data/ysj/dataset/LF_video_crop_half/ambushfight_1'
    view_dirs = get_view_dirs(imgdir)
    view_paths = [os.path.join(imgdir, view_dir) for view_dir in view_dirs]
    print(get_imgs_from_view_dirs(view_paths, 20))

if __name__ == '__main__':
    __main__()