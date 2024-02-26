import torch
from functools import partial
from torchvision.models import efficientnet_b0
from torchvision.models.resnet import resnet18
from torchvision.models.convnext import convnext_tiny
from kornia.geometry.epipolar import triangulate_points
from src.util.cam_geometry import project_to_relative_coordinates


class DenseRegressor(torch.nn.Module):
    def __init__(self, name="EfficientNet", dim_output=2, num_point=26317, loss_name='L1'):
        super().__init__()
        self.name = name
        self.dim_output=dim_output
        self.num_point=num_point
        self.loss_name=loss_name
        self.dense_regressor = self.create_regressor()
        self.loss_function = self.create_loss()
    
    def create_regressor(self):
        if self.name == 'ConvNext':
            model = convnext_tiny(num_classes=self.dim_output*self.num_point)
        elif self.name == 'ResNet18':
            model = resnet18(num_classes=self.dim_output*self.num_point)
        else:
            model = efficientnet_b0(weights='DEFAULT', num_classes=self.dim_output*self.num_point)
        return model
    
    def create_loss(self):
        if self.loss_name == 'L1':
            return torch.nn.L1Loss()
        else:
            raise ValueError('Loss function name incorrect.')
    
    def forward(self, x):
        return self.dense_regressor(x)
    
    def calc_losses(self, batch):
        image=batch["image"] # (2, 3, 256, 256)
        target_kpts=batch["target_keypoints"] # (2, 26317, 2)
        b, np, d = target_kpts.shape
        
        pred_kpts = self.forward(image) # (2, 26317*2)
        pred_kpts = pred_kpts.view(b, np, d) # (2, 26317, 2)
        
        loss_dict = {}
        loss_dict["total"] = self.loss_function(target_kpts, pred_kpts)
        
        return loss_dict