import torch
from . import _C

class gridinterpole_1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid: torch.Tensor, rays: torch.Tensor):
        """
        grid: [C, U]
        rays: [N, 1]
        """
        N = rays.shape[0]
        C = grid.shape[0]
        out = torch.empty((N, C), dtype=grid.dtype, device=grid.device)

        # forward 호출
        _C.forward_1d(grid, rays, out)
        
        # backward에 필요한 텐서 저장
        ctx.save_for_backward(rays)
        ctx.grid_shape = grid.shape  # (C, U)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: [N, C]
        returns: dgrid, drays(없으면 None)
        """
        (rays,) = ctx.saved_tensors 
        (C, U) = ctx.grid_shape

        # grid에 대한 grad
        grad_grid = torch.zeros((C, U), dtype=grad_output.dtype, device=grad_output.device)

        # 실제 backward cuda 호출
        grad_output = grad_output.contiguous()
        _C.backward_1d(grad_output, rays, grad_grid)
        return grad_grid, None
    
class gridinterpole_2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid: torch.Tensor, rays: torch.Tensor):
        """
        grid: [C, U, V]
        rays: [N, 2]
        """
        N = rays.shape[0]
        C = grid.shape[0]
        out = torch.empty((N, C), dtype=grid.dtype, device=grid.device)

        # forward 호출
        _C.forward_2d(grid, rays, out)
        
        # backward에 필요한 텐서 저장
        ctx.save_for_backward(rays)
        ctx.grid_shape = grid.shape  # (C, U, V)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: [N, C]
        returns: dgrid, drays(없으면 None)
        """
        (rays,) = ctx.saved_tensors
        (C, U, V) = ctx.grid_shape

        # grid에 대한 grad
        grad_grid = torch.zeros((C, U, V), dtype=grad_output.dtype, device=grad_output.device)

        # 실제 backward cuda 호출
        grad_output = grad_output.contiguous()
        _C.backward_2d(grad_output, rays, grad_grid)
        return grad_grid, None
    
class gridinterpole_3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid: torch.Tensor, rays: torch.Tensor):
        """
        grid: [C, U, V, S]
        rays: [N, 3]
        """
        N = rays.shape[0]
        C = grid.shape[0]
        out = torch.empty((N, C), dtype=grid.dtype, device=grid.device)

        # forward 호출  
        _C.forward_3d(grid, rays, out)

        # backward에 필요한 텐서 저장
        ctx.save_for_backward(rays)
        ctx.grid_shape = grid.shape  # (C, U, V, S)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: [N, C]
        returns: dgrid, drays(없으면 None)
        """
        (rays,) = ctx.saved_tensors
        (C, U, V, S) = ctx.grid_shape

        # grid에 대한 grad
        grad_grid = torch.zeros((C, U, V, S), dtype=grad_output.dtype, device=grad_output.device)

            # 실제 backward cuda 호출
        grad_output = grad_output.contiguous()
        _C.backward_3d(grad_output, rays, grad_grid)
        return grad_grid, None
    

class gridinterpole_4d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid: torch.Tensor, rays: torch.Tensor):
        """
        grid: [C, U, V, S, T]
        rays: [N, 4], in [0,1]
        returns: [N, C]
        """
        N = rays.shape[0]
        C = grid.shape[0]
        out = torch.empty((N, C), dtype=grid.dtype, device=grid.device)

        # forward 호출
        _C.forward_4d(grid, rays, out)

        # backward에 필요한 텐서 저장
        ctx.save_for_backward(rays)
        ctx.grid_shape = grid.shape  # (C, U, V, S, T)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: [N, C]
        returns: dgrid, drays(없으면 None)
        """
        (rays,) = ctx.saved_tensors
        (C, U, V, S, T) = ctx.grid_shape

        # grid에 대한 grad
        grad_grid = torch.zeros((C, U, V, S, T), dtype=grad_output.dtype, device=grad_output.device)

        # 실제 backward cuda 호출
        grad_output = grad_output.contiguous()
        _C.backward_4d(grad_output, rays, grad_grid)
        return grad_grid, None
    
def grid_interpolate_1d(grid: torch.Tensor, rays: torch.Tensor):
    """
    편의 함수:
    grid: [C, U]
    rays: [N, 1]  (normalized coords)
    -> returns [N, C]
    """
    return gridinterpole_1d.apply(grid, rays)

def grid_interpolate_2d(grid: torch.Tensor, rays: torch.Tensor):
    """
    편의 함수:
    grid: [C, U, V]
    rays: [N, 2]  (normalized coords)
    -> returns [N, C]
    """
    return gridinterpole_2d.apply(grid, rays)

def grid_interpolate_3d(grid: torch.Tensor, rays: torch.Tensor):
    """
    편의 함수:
    grid: [C, U, V, S]
    rays: [N, 3]  (normalized coords)
    -> returns [N, C]
    """
    return gridinterpole_3d.apply(grid, rays)
    
def grid_interpolate_4d(grid: torch.Tensor, rays: torch.Tensor):
    """
    편의 함수:
    grid: [C, U, V, S, T]
    rays: [N, 4]  (normalized coords)
    -> returns [N, C]
    """
    return gridinterpole_4d.apply(grid, rays)

def total_variation_add_grad(param: torch.Tensor, grad: torch.Tensor, wx: float, wy: float, wz: float, wt: float, dimension: int, dense_mode: bool):
    
    return _C.total_variation_add_grad(param, grad, wx, wy, wz, wt, dimension, dense_mode)