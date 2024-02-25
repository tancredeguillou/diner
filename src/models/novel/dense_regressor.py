import torch
from functools import partial
# from torchvision.models.efficientnet import _efficientnet, MBConvConfig
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
            # bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.0)
            # inverted_residual_setting = [
            #     bneck_conf(1, 3, 1, 32, 16, 1),
            #     bneck_conf(4, 3, 2, 16, 32, 2),
            #     bneck_conf(4, 3, 2, 32, 48, 2),
            #     bneck_conf(4, 3, 2, 48, 96, 3),
            #     bneck_conf(6, 3, 1, 96, 112, 5),
            #     bneck_conf(6, 3, 2, 112, 192, 8),
            # ]
            # last_channel=None
            # model = _efficientnet(
            #     inverted_residual_setting=inverted_residual_setting,
            #     dropout=0.2,
            #     last_channel=last_channel,
            #     pretrained=None,
            #     progress=True,
            #     width_mult=1.0,
            #     depth_mult=1.0,
            #     arch="efficientnet_b0",
            #     num_classes=self.dim_output*self.num_point)
            model = efficientnet_b0(weights='DEFAULT', num_classes=self.dim_output*self.num_point)
            
#             inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
#     dropout: float,
#     last_channel: Optional[int],
#     weights: Optional[WeightsEnum],
#     progress: bool,
#     **kwargs: Any,

        return model
    
    def create_loss(self):
        if self.loss_name == 'L1':
            return torch.nn.L1Loss()
        else:
            raise ValueError('Loss function name incorrect.')
    
    def forward(self, x):
        return self.dense_regressor(x)
    
    def calc_losses(self, batch):
        images=batch["src_rgbs"]
        b, n, c, h, w = images.shape
        vertices=batch["src_vertices"]
        extrinsics=batch["src_extrinsics"][..., :-1, :]
        intrinsics=batch["src_intrinsics"]
        # print('extrinsics', extrinsics.shape, flush=True)
        # print('intrinsics', intrinsics.shape, flush=True)
        
        projection_matrices = torch.matmul(intrinsics, extrinsics)
        # print('projection_matrics', projection_matrices.shape, flush=True)
        
        output = self.forward(images.view(b*n, c, h, w)).view(b, n, self.num_point, self.dim_output)
        # print('output shape', output.shape, flush=True)
        
        proj_split = projection_matrices.split(1, dim=1)
        # print('proj_split shape', len(proj_split), flush=True)
        # print('proj_list 0 shape', proj_split[0].shape, flush=True)
        output_split = output.split(1, dim=1)
        # print('output_split shape', len(output_split), flush=True)
        # print('output_split 0 shape', output_split[0].shape, flush=True)
        
        final_vertices = triangulate_points(proj_split[0], proj_split[1], output_split[0], output_split[1])
        # print('final vertices shape', final_vertices.shape, flush=True)
        # print('vertices shape', vertices.shape, flush=True)
        
        loss_dict = {}
        loss_dict["total"] = self.loss_function(vertices, final_vertices.squeeze())
        
        # raise ValueError()
        return loss_dict