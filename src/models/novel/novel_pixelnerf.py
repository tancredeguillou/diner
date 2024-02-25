"""
Code heavily inspired by https://github.com/sxyu/pixel-nerf
"""

from torchvision.transforms import Normalize
import torch
import torch.nn.functional as F
from src.util.import_helper import import_obj
from src.models.positional_encoding import PositionalEncoding
from src.util.depth2normal import depth2normal
from src.util.torch_helpers import index_depth


class PixelNeRF(torch.nn.Module):
    def __init__(self, poscode_conf, encoder_conf, mlp_fine_conf):
        super().__init__()
        self.poscode = PositionalEncoding(**poscode_conf.kwargs, d_in=3)
        self.depthcode = PositionalEncoding(**poscode_conf.kwargs, d_in=1)
        self.encoder = import_obj(encoder_conf.module)(**encoder_conf.kwargs)
        self.d_in = self.poscode.d_out + self.depthcode.d_out + 3
        self.d_latent = self.encoder.latent_size # + 1
        self.d_out = 4
        self.mlp_fine = import_obj(mlp_fine_conf.module)(**mlp_fine_conf.kwargs,
                                                         d_latent=self.d_latent,
                                                         d_in=self.d_in,
                                                         d_out=self.d_out)

        # Setting buffers for reference camera parameters
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("image_shape", torch.empty(2), persistent=False)
        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        self.register_buffer("c", torch.empty(1, 2), persistent=False)
        
        # Setting buffers for general camera parameters
        self.register_buffer("gen_poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("gen_image_shape", torch.empty(2), persistent=False)
        self.register_buffer("gen_focal", torch.empty(1, 2), persistent=False)
        self.register_buffer("gen_c", torch.empty(1, 2), persistent=False)
        
        # Setting buffers for target camera parameters
        # self.register_buffer("target_poses", torch.empty(1, 3, 4), persistent=False)
        # self.register_buffer("target_image_shape", torch.empty(2), persistent=False)
        # self.register_buffer("target_focal", torch.empty(1, 2), persistent=False)
        # self.register_buffer("target_c", torch.empty(1, 2), persistent=False)
        
        self.gen_latent = torch.nn.Parameter(torch.randn(512, 192, 192))
        self.gen_latent.requires_grad = True
        
        # for name, param in self.named_parameters():
        #     if param.requires_grad and 'gen_latent' in name:
        #         print(name, param.data, flush=True)

        self.normalize_rgb = Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])

    def encode(self, images, depths, depths_std, extrinsics, intrinsics):
        """
        creates and stores feature maps, call encode() before using forward()!
        @param images: SB, NV, 3, H, W
        @param depths: SB, NV, 1, H, W
        @param extrinsics: SB, NV, 4, 4
        @param intrinsics: SB, NV, 3, 3
        @return:
        """
        images = self.normalize_rgb(images)
        normals = depth2normal(depths.flatten(end_dim=1), intrinsics.flatten(end_dim=1)).reshape_as(images)
        self.encoder(images, depths, depths_std, normals)
        self.poses = extrinsics
        self.c = intrinsics[:, :, :2, -1]
        self.focal = intrinsics[:, :, torch.tensor([0, 1]), torch.tensor([0, 1])]
        self.image_shape[0] = images.shape[-1]  # Width
        self.image_shape[1] = images.shape[-2]  # Height

        return
    
    def encode_gen(self, extrinsics, intrinsics, image_shape):
        self.gen_poses = extrinsics.unsqueeze(1)
        self.gen_c = intrinsics[:, :2, -1].unsqueeze(1)
        self.gen_focal = intrinsics[:, torch.tensor([0, 1]), torch.tensor([0, 1])].unsqueeze(1)
        self.gen_image_shape[0] = image_shape[-1]  # Width
        self.gen_image_shape[1] = image_shape[-2]  # Height
        
        return
    
#     def encode_target(self, target_image, target_depth, target_depth_std, target_extrinsic, target_intrinsic):
#         """
#         creates and stores feature maps, call encode() before using forward()!
#         @param images: SB, 1, 3, H, W
#         @param depths: SB, 1, 1, H, W
#         @param extrinsics: SB, 1, 4, 4
#         @param intrinsics: SB, 1, 3, 3
#         @return:
#         """
#         target_image = self.normalize_rgb(target_image)
#         target_normal = depth2normal(target_depth.flatten(end_dim=1), target_intrinsic.flatten(end_dim=1)).reshape_as(target_image)
#         # self.encoder(images, depths, depths_std, normals)
#         self.target_depth = target_depth
#         self.target_depth_std = target_depth_std
#         self.target_normal = target_normal
#         self.target_poses = target_extrinsic
#         self.target_c = target_intrinsic[:, :, :2, -1]
#         self.target_focal = target_intrinsic[:, :, torch.tensor([0, 1]), torch.tensor([0, 1])]
#         self.target_image_shape[0] = target_image.shape[-1]  # Width
#         self.target_image_shape[1] = target_image.shape[-2]  # Height

#         return

    def index(self, uv):
        SB, NV, N, _ = uv.shape
        latent = self.gen_latent.repeat(SB, NV, 1, 1, 1)
        # print('gen latent repeat shape', latent.shape, flush=True)
        
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (SB, NV, N, 2) image points (x,y) in interval [-1, -1] (topleft pixelcorner)
                                                             to [1,1] (bottomright pixelcorner)
        :return (SB, NV, L, N) L is latent size
        """

        assert uv.shape[:2] == latent.shape[:2]

        N_ = SB * NV
        uv = uv.view(N_, N, 2)
        latent = latent.view(N_, *latent.shape[-3:])

        # correcting uv for feature padding
        latent_size = torch.tensor([latent.shape[-1], latent.shape[-2]], device=uv.device)  # W+pad, H+pad
        uv = uv * ((latent_size - self.encoder.feature_padding * 2) / latent_size).view(1, 1, 2)

        uv = uv.unsqueeze(2)  # (B, N, 1, 2)
        samples = F.grid_sample(
            latent,
            uv,
            align_corners=False,
            mode=self.encoder.index_interp,
            padding_mode=self.encoder.index_padding,
        )
        samples = samples[:, :, :, 0]  # (N_, C, N)

        samples = samples.view(SB, NV, *samples.shape[-2:])  # (SB, NV, C, N)
        return samples

    def forward(self, xyz, gen_xyz, viewdirs): # tgt_xyz
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :param viewdirs (SB, B, 3)
        :return (SB, B, 4) r g b sigma
        """
        SB, B, _ = xyz.shape
        NV = self.encoder.nviews
        assert SB == self.encoder.nobjects

        # Transform query points into the camera spaces of the input views
        xyz = xyz.unsqueeze(1).expand(-1, NV, -1, -1)  # (SB, NV, B, 3)
        xyz_rot = torch.matmul(self.poses[:, :, :3, :3], xyz.transpose(-2, -1)).transpose(-2, -1)
        xyz = xyz_rot + self.poses[:, :, :3, -1].unsqueeze(-2)  # (SB, NV, B, 3)
        
        gen_xyz = gen_xyz.unsqueeze(1).expand(-1, NV, -1, -1)  # (SB, NV, B, 3)
        gen_xyz_rot = torch.matmul(self.gen_poses[:, :, :3, :3], gen_xyz.transpose(-2, -1)).transpose(-2, -1)
        gen_xyz = gen_xyz_rot + self.gen_poses[:, :, :3, -1].unsqueeze(-2)  # (SB, NV, B, 3)

        # Positional encoding (no viewdirs)
        z_feature = self.poscode(xyz)  # SB, NV, B, d_pos

        # add viewdirs
        viewdirs = viewdirs.unsqueeze(1).expand(-1, NV, -1, -1)  # (SB, NV, B, 3)
        viewdirs = torch.matmul(self.poses[:, :, :3, :3],
                                viewdirs.transpose(-1, -2)).transpose(-1, -2)  # (SB, NV, B, 3)
        z_feature = torch.cat((z_feature, viewdirs), dim=-1)  # (SB, NV, B, d_in)

        # Grab encoder's latent code.
        uv = xyz[..., :2] / xyz[..., 2:]  # (SB, NV, B, 2)
        uv *= self.focal.unsqueeze(-2)
        uv += self.c.unsqueeze(-2)
        uv = uv / self.image_shape * 2 - 1  # assumes outer edges of pixels correspond to uv coordinates -1 / 1
        # print('uv', uv.shape, uv.min(), uv.max(), flush=True)
        
        gen_uv = gen_xyz[..., :2] / gen_xyz[..., 2:]
        gen_uv *= self.gen_focal.unsqueeze(-2)
        gen_uv += self.gen_c.unsqueeze(-2)
        gen_uv = gen_uv / self.gen_image_shape * 2 - 1  # assumes outer edges of pixels correspond to uv coordinates -1 / 1

        latent = self.encoder.index(uv)  # (SB, NV, latent, B)
        latent = latent.transpose(-1, -2)  # (SB, NV, B, latent)
        
        # print('gen uv', gen_uv.shape, gen_uv.min(), gen_uv.max(), flush=True)
        gen_latent = self.index(gen_uv)
        gen_latent = gen_latent.transpose(-1, -2)
        # print('gen latent shape 2', gen_latent.shape, flush=True)
        
        final_latent = gen_latent + latent

        # encoding dist2depth
        ref_depth = self.encoder.index_depth(uv)  # SB, NV, 1, B
        depth_dist = ref_depth.squeeze(-2) - xyz[..., -1]  # (SB, NV, B)
        depth_feature = self.depthcode(depth_dist.unsqueeze(-1))  # (SB, NV, B, C)

        # # visualizing sampled points and mapped depths (*-1 and clipped to 1.)
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # ax.scatter(xyz[0, 0, :, 0].detach().cpu(), xyz[0, 0, :, 1].detach().cpu(), xyz[0, 0, :, 2].detach().cpu(), s=.1)
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # plt.show()

        # (SB, NV, B, 1) - batch, nviews, we get B inputs, for each input get the depth value
        # tgt_xyz = tgt_xyz.unsqueeze(1)  # (SB, 1, B, 3)
        # tgt_xyz_rot = torch.matmul(self.target_poses[:, :, :3, :3], tgt_xyz.transpose(-2, -1)).transpose(-2, -1)
        # tgt_xyz = tgt_xyz_rot + self.target_poses[:, :, :3, -1].unsqueeze(-2)  # (SB, NV, B, 3)
        # tgt_uv = tgt_xyz[..., :2] / tgt_xyz[..., 2:]  # (SB, NV, B, 2)
        # tgt_uv *= self.target_focal.unsqueeze(-2)
        # tgt_uv += self.target_c.unsqueeze(-2)
        # tgt_uv = tgt_uv / self.target_image_shape * 2 - 1  # assumes outer edges of pixels correspond to uv coordinates -1 / 1
        # # tgt_depth = index_depth(self.target_depth, tgt_uv)  # SB, NV, 1, B
        # tgt_depth = tgt_depth.transpose(-1, -2)
        # tgt_depth = tgt_depth.expand(-1, NV, -1, -1)
        
        mlp_input = torch.cat((final_latent, z_feature, depth_feature), dim=-1)  # (SB, NV, B, C_in) # tgt_depth, positionned after latent

        # Run main NeRF network
        mlp_output = self.mlp_fine(
            mlp_input,
            combine_dim=1
        )

        # Interpret the output
        mlp_output = mlp_output.reshape(SB, B, self.d_out)

        rgb = mlp_output[..., :3]
        sigma = mlp_output[..., 3:4]

        output_list = [torch.sigmoid(rgb), torch.relu(sigma)]
        output = torch.cat(output_list, dim=-1)

        return output  # (SB, B, 4)