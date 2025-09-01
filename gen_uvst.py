import torch

def get_uvst(x, y, H, W, uv_scale=1, st_scale=0.1):
    aspect = W / H
    u = torch.linspace(-1, 1, W, dtype=torch.float32)
    v = torch.linspace(-1, 1, H, dtype=torch.float32) / aspect
    v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
    
    u = u_grid * uv_scale
    v = v_grid * uv_scale
    
    s = torch.ones_like(u_grid) * x * st_scale
    t = torch.ones_like(v_grid) * y * st_scale
    uvst = torch.stack([u, v, s, t], dim=-1)
    
    return uvst

def get_uvst_time(x, y, tt, H, W, num_frames=50, uv_scale=1, st_scale=0.1):
    aspect = W / H
    u = torch.linspace(-1, 1, W, dtype=torch.float32)
    v = torch.linspace(-1, 1, H, dtype=torch.float32) / aspect
    v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
    
    u = u_grid * uv_scale
    v = v_grid * uv_scale
    
    s = torch.ones_like(u_grid) * x * st_scale
    t = torch.ones_like(v_grid) * y * st_scale
    
    time = tt / (num_frames - 1)
    time = torch.ones_like(u_grid) * time
    uvst = torch.stack([u, v, s, t, time], dim=-1)
    
    return uvst

def __main__():
    # test
    uvst = get_uvst(0, 0, 1024, 1024, 1, 0.1)
    print(uvst.shape)
    print(uvst)
    uvst_time = get_uvst_time(0, 0, 3, 1024, 1024, 10, 1, 0.1)
    print(uvst_time.shape)
    print(uvst_time)

if __name__ == "__main__":
    __main__()