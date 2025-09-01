#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void total_variation_add_grad_cuda(torch::Tensor param, torch::Tensor grad, float wx, float wy, float wz, float wt, int dimension, bool dense_mode);

void grid_interpole_1d_forward_cuda(
    torch::Tensor grid,
    torch::Tensor rays,
    torch::Tensor output
);

void grid_interpole_1d_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor rays,
    torch::Tensor grad_grid
);

void grid_interpole_1d_forward(
    torch::Tensor grid,
    torch::Tensor rays,
    torch::Tensor output
) {
    grid_interpole_1d_forward_cuda(grid, rays, output);
}

void grid_interpole_1d_backward(
    torch::Tensor grad_out,
    torch::Tensor rays,
    torch::Tensor grad_grid
) {
    grid_interpole_1d_backward_cuda(grad_out, rays, grad_grid);
}

torch::Tensor grid_interpolate_1d(
    torch::Tensor grid,
    torch::Tensor rays
) {
    TORCH_CHECK(grid.dim() == 2, "grid must be 2D tensor");
    TORCH_CHECK(rays.dim() == 2 && rays.size(1) == 1, "rays must be of shape [N, 1]");
    auto output = torch::zeros({rays.size(0), grid.size(1)}, grid.options());
    grid_interpole_1d_forward_cuda(grid, rays, output);
    return output;
}

torch::Tensor grid_interpolate_1d_backward(
    torch::Tensor grad_out,
    torch::Tensor rays,
    std::vector<int64_t> grid_size
) {
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must be 2D tensor");
    TORCH_CHECK(rays.dim() == 2 && rays.size(1) == 1, "rays must be of shape [N, 1]");
    auto grad_grid = torch::zeros({grad_out.size(1), grid_size[0]}, grad_out.options());
    grid_interpole_1d_backward_cuda(grad_out, rays, grad_grid);
    return grad_grid;
}

void grid_interpole_2d_forward_cuda(
    torch::Tensor grid,
    torch::Tensor rays,
    torch::Tensor output
);

void grid_interpole_2d_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor rays,
    torch::Tensor grad_grid
);

void grid_interpole_2d_forward(
    torch::Tensor grid,
    torch::Tensor rays,
    torch::Tensor output
) {
    grid_interpole_2d_forward_cuda(grid, rays, output);
}

void grid_interpole_2d_backward(
    torch::Tensor grad_out,
    torch::Tensor rays,
    torch::Tensor grad_grid
) {
    grid_interpole_2d_backward_cuda(grad_out, rays, grad_grid);
}

torch::Tensor grid_interpolate_2d(
    torch::Tensor grid,
    torch::Tensor rays
) {
    TORCH_CHECK(grid.dim() == 3, "grid must be 3D tensor");
    TORCH_CHECK(rays.dim() == 2 && rays.size(1) == 2, "rays must be of shape [N, 2]");
    auto output = torch::zeros({rays.size(0), grid.size(0)}, grid.options());
    grid_interpole_2d_forward_cuda(grid, rays, output);
    return output;
}

torch::Tensor grid_interpolate_2d_backward(
    torch::Tensor grad_out,
    torch::Tensor rays,
    std::vector<int64_t> grid_size
) {
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must be 2D tensor");
    TORCH_CHECK(rays.dim() == 2 && rays.size(1) == 2, "rays must be of shape [N, 2]");
    auto grad_grid = torch::zeros({grad_out.size(1), grid_size[0], grid_size[1]}, grad_out.options());
    grid_interpole_2d_backward_cuda(grad_out, rays, grad_grid);
    return grad_grid;
}

void grid_interpole_3d_forward_cuda(
    torch::Tensor grid,
    torch::Tensor rays,
    torch::Tensor output
);

void grid_interpole_3d_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor rays,
    torch::Tensor grad_grid
);

void grid_interpole_3d_forward(
    torch::Tensor grid,
    torch::Tensor rays,
    torch::Tensor output
) {
    grid_interpole_3d_forward_cuda(grid, rays, output);
}

void grid_interpole_3d_backward(
    torch::Tensor grad_out,
    torch::Tensor rays,
    torch::Tensor grad_grid
) {
    grid_interpole_3d_backward_cuda(grad_out, rays, grad_grid);
}

torch::Tensor grid_interpolate_3d(
    torch::Tensor grid,
    torch::Tensor rays
) {
    TORCH_CHECK(grid.dim() == 4, "grid must be 4D tensor");
    TORCH_CHECK(rays.dim() == 2 && rays.size(1) == 3, "rays must be of shape [N, 3]");
    auto output = torch::zeros({rays.size(0), grid.size(0)}, grid.options());
    grid_interpole_3d_forward_cuda(grid, rays, output);
    return output;
}

torch::Tensor grid_interpolate_3d_backward(
    torch::Tensor grad_out,
    torch::Tensor rays,
    std::vector<int64_t> grid_size
) {
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must be 2D tensor");
    TORCH_CHECK(rays.dim() == 2 && rays.size(1) == 3, "rays must be of shape [N, 3]");
    auto grad_grid = torch::zeros({grad_out.size(1), grid_size[0], grid_size[1], grid_size[2]}, grad_out.options());
    grid_interpole_3d_backward_cuda(grad_out, rays, grad_grid);
    return grad_grid;
}

void grid_interpole_4d_forward_cuda(
    torch::Tensor grid,
    torch::Tensor rays,
    torch::Tensor output
);

void grid_interpole_4d_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor rays,
    torch::Tensor grad_grid
);

void grid_interpole_4d_forward(
    torch::Tensor grid,
    torch::Tensor rays,
    torch::Tensor output
) {
    grid_interpole_4d_forward_cuda(grid, rays, output);
}

void grid_interpole_4d_backward(
    torch::Tensor grad_out,
    torch::Tensor rays,
    torch::Tensor grad_grid
) {
    grid_interpole_4d_backward_cuda(grad_out, rays, grad_grid);
}

// 4D 그리드 보간 forward 함수
torch::Tensor grid_interpolate_4d(
    torch::Tensor grid,
    torch::Tensor rays
) {
    // 입력 텐서의 크기 확인
    TORCH_CHECK(grid.dim() == 5, "grid must be 5D tensor");
    TORCH_CHECK(rays.dim() == 2 && rays.size(1) == 4, "rays must be of shape [N, 4]");
    
    // 출력 텐서 생성
    auto output = torch::zeros({rays.size(0), grid.size(0)}, grid.options());
    
    // CUDA 함수 호출
    grid_interpole_4d_forward_cuda(grid, rays, output);
    
    return output;
}

torch::Tensor grid_interpolate_4d_backward(
    torch::Tensor grad_out,
    torch::Tensor rays,
    std::vector<int64_t> grid_size
) {
    // 입력 텐서의 크기 확인
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must be 2D tensor");
    TORCH_CHECK(rays.dim() == 2 && rays.size(1) == 4, "rays must be of shape [N, 4]");
    
    // 출력 텐서 생성
    auto grad_grid = torch::zeros({grad_out.size(1), grid_size[0], grid_size[1], grid_size[2], grid_size[3]}, 
                                 grad_out.options());
    
    // CUDA 함수 호출
    grid_interpole_4d_backward_cuda(grad_out, rays, grad_grid);
    
    return grad_grid;
}

void total_variation_add_grad(torch::Tensor param, torch::Tensor grad, float wx, float wy, float wz, float wt, int dimension, bool dense_mode) {
  total_variation_add_grad_cuda(param, grad, wx, wy, wz, wt, dimension, dense_mode);
}

// pybind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_1d", &grid_interpole_1d_forward, "1D grid interpolate forward (CUDA)");
    m.def("backward_1d", &grid_interpole_1d_backward, "1D grid interpolate backward (CUDA)");
    m.def("grid_interpolate_1d", &grid_interpolate_1d, "1D grid interpolation (CUDA)");
    m.def("grid_interpolate_1d_backward", &grid_interpolate_1d_backward, "Backward pass for 1D grid interpolation (CUDA)");

    m.def("forward_2d", &grid_interpole_2d_forward, "2D grid interpolate forward (CUDA)");
    m.def("backward_2d", &grid_interpole_2d_backward, "2D grid interpolate backward (CUDA)");
    m.def("grid_interpolate_2d", &grid_interpolate_2d, "2D grid interpolation (CUDA)");
    m.def("grid_interpolate_2d_backward", &grid_interpolate_2d_backward, "Backward pass for 2D grid interpolation (CUDA)");

    m.def("forward_3d", &grid_interpole_3d_forward, "3D grid interpolate forward (CUDA)");
    m.def("backward_3d", &grid_interpole_3d_backward, "3D grid interpolate backward (CUDA)");
    m.def("grid_interpolate_3d", &grid_interpolate_3d, "3D grid interpolation (CUDA)");
    m.def("grid_interpolate_3d_backward", &grid_interpolate_3d_backward, "Backward pass for 3D grid interpolation (CUDA)");

    m.def("forward_4d", &grid_interpole_4d_forward, "4D grid interpolate forward (CUDA)");
    m.def("backward_4d", &grid_interpole_4d_backward, "4D grid interpolate backward (CUDA)");
    m.def("grid_interpolate_4d", &grid_interpolate_4d, "4D grid interpolation (CUDA)");
    m.def("grid_interpolate_4d_backward", &grid_interpolate_4d_backward, "Backward pass for 4D grid interpolation (CUDA)");

    m.def("total_variation_add_grad", &total_variation_add_grad, "Total variation add grad (CUDA)");
}
