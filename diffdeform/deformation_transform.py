import torch
from torch import nn
import torch.nn.functional as F

class DeformationTransform(nn.Module):
    def __init__(self, K, L, image_size, mode = "zeros"):
        super(DeformationTransform, self).__init__()
        self.mode = mode
        self.K = K
        self.L = L
        self.image_size = image_size
        self.control_point_displacements = nn.Parameter(5*torch.randn(2, K, L))

    def create_dense_displacement_field(self, displacements):
        # K, L = displacements.shape[1], displacements.shape[2]
        K = self.K
        L = self.L
        height, width = self.image_size

        # Create a grid of pixel coordinates
        pixel_grid_y, pixel_grid_x = torch.meshgrid(
            torch.arange(height, device=displacements.device),
            torch.arange(width, device=displacements.device),
            indexing='ij'
        )
        pixel_grid = torch.stack([pixel_grid_x, pixel_grid_y], dim=-1)  # Shape: (height, width, 2)

        # Normalize pixel grid to [-1, 1]
        pixel_grid = 2.0 * (pixel_grid / torch.tensor([width - 1, height - 1], device=displacements.device)) - 1.0
        pixel_grid = pixel_grid.unsqueeze(0)  # Shape: (1, height, width, 2)
        
        # Interpolate displacements to create dense displacement field
        dense_displacement_field = F.grid_sample(
            displacements.unsqueeze(0),  # Shape: (1, 2, K, L)
            pixel_grid,  # Shape: (1, height, width, 2)
            mode='bicubic',  # Use bicubic interpolation
            align_corners=True
        )
        
        return dense_displacement_field # Shape : (1, 2, height, width)

    def apply_displacement_field(self, image, displacement_field):
        # Convert image and displacement_field to float if not already
        image = image.float()
        displacement_field = displacement_field.float()
        
        # Create a grid of coordinates
        N, C, H, W = image.shape
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=image.device),
            torch.arange(W, device=image.device),
            indexing='ij'
        )
        pixel_grid = torch.stack([x_coords, y_coords], dim=-1).float()  # Shape: (H, W, 2)
        pixel_grid = pixel_grid + displacement_field.squeeze().permute(1,2,0)
        
        # Normalize pixel grid to [-1, 1]
        pixel_grid = 2.0 * (pixel_grid / torch.tensor([W - 1, H - 1], device=image.device)) - 1.0
        pixel_grid = pixel_grid.unsqueeze(0).repeat(N, 1, 1, 1)  # Shape: (N, H, W, 2)

        # Apply displacement field
        deformed_image = F.grid_sample(
            image,  # Shape: (N, C, H, W)
            pixel_grid,
            mode='bilinear',
            align_corners=True,
            padding_mode = self.mode
        )

        return deformed_image

    def forward(self, image):
        dense_displacement_field = self.create_dense_displacement_field(self.control_point_displacements)
        deformed_image = self.apply_displacement_field(image, dense_displacement_field)
        return deformed_image

