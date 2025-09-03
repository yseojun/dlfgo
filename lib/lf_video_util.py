import os


def get_lf_video_validation_idx(grid_num, frame_num):
    result = []
    mid = grid_num // 2
    min = mid // 2
    max = mid + min
    
    val_idx = [min, mid, max]
    
    for y in val_idx:
        for x in val_idx:
            for f in range(frame_num):
                idx = (y * grid_num + x) * frame_num + f
                result.append(idx)
    print("validation index : ", result)
    return result

def get_frame_from_idx(idx, grid_num, frame_num):
    y = idx // (grid_num * frame_num)
    x = (idx % (grid_num * frame_num)) // frame_num
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