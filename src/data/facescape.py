from pathlib import Path
import os
import numpy as np
import torch
import time
import json
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image
from src.util.torch_helpers import dict_2_torchdict
import itertools
from itertools import product
import tqdm
from src.util.cam_geometry import to_homogeneous_trafo

OPENCV2OPENGL = np.array([[1., 0., 0., 0.], [0., -1., 0., 0], [0., 0., -1., 0.], [0., 0., 0., 1.]], dtype=np.float32)


class FacescapeDataSet(torch.utils.data.Dataset):
    znear = 1.
    zfar = 2.5
    RGBA_FNAME = "rgba_colorcalib_v2.png"
    DEPTH_FNAME = "depth_gt_pred_conf.png"
    DEPTH_MESH_FNAME = "depth_mesh.png"

    def __init__(self, model, root: Path, stage, range_hor=45, range_vert=30, slide_range=40, slide_step=20, depth_type='original', depth_fname=None):
        """
        Capstudio Data Loading Class.
        Camera extrinsics follow OPENCV convention (cams look in positive z direction, y points downwards)
        :param root:
        :param stage:
        :param banned_views:
        :param kwargs:
        """
        super().__init__()
        self.model = model
        assert os.path.exists(root)
        self.data_dir = Path(root)
        self.stage = stage
        self.rnd = np.random.default_rng() if stage == "train" else np.random.default_rng(128)
        self.DEPTH_FNAME = depth_fname if depth_fname is not None else self.DEPTH_FNAME
        self.DEPTH_MESH_FNAME = self.DEPTH_MESH_FNAME
        self.range_hor = range_hor
        self.range_vert = range_vert
        self.nsource = 2
        self.slide_range = slide_range
        self.slide_step = slide_step
        print('DEPTH_TYPE = ', depth_type, flush=True)
        self.depth_type = depth_type
        #self.DEPTH_STD_FNAME = self.DEPTH_FNAME.replace(".png", "_conf.png")
        self.conf2std = self._getconf2std()
        self.metas = self.get_metas()

    def _getconf2std(self):
        conf2std = lambda x: -1.582e-2 * x + 1.649e-2
        return conf2std

    @staticmethod
    def read_rgba(p: Path, symmetric_range=False, bg=1.):
        rgb, a = torch.split(pil_to_tensor(Image.open(p)).float() / 255., (3, 1))  # range: 0 ... 1

        if symmetric_range:  # rgb range: -1 ... 1
            rgb = rgb * 2 - 1

        rgb.permute(1, 2, 0)[a[0] < .5] = bg
        return rgb, a

    @staticmethod
    def read_depth(p: Path, mp: Path, depth_type='original'):
        UINT16_MAX = 65535
        SCALE_FACTOR = 1e-4
        
        unprocessed_depth_img = Image.open(p)
        unprocessed_mesh_depth_img = Image.open(mp)
        
        pred_img = pil_to_tensor(unprocessed_mesh_depth_img).float() * SCALE_FACTOR
        conf_img = torch.where(pred_img == float(0.),
                               float(0.),
                               float(0.8))
        
        width = unprocessed_depth_img.width // 3

        gt_pil = unprocessed_depth_img.crop((0, 0, width, unprocessed_depth_img.height))
        pred_pil = unprocessed_depth_img.crop((width, 0, 2 * width, unprocessed_depth_img.height))
        conf_pil = unprocessed_depth_img.crop((2 * width, 0,
                                               unprocessed_depth_img.width,
                                               unprocessed_depth_img.height))
        
        pred_MVS_img = pil_to_tensor(pred_pil).float() * SCALE_FACTOR
        conf_MVS_img = pil_to_tensor(conf_pil).float() * SCALE_FACTOR
        
        if depth_type == 'original':
            return pred_MVS_img, conf_MVS_img
        elif depth_type == 'mesh':
            return pred_img, conf_img
        elif depth_type == 'merge':
            pred_img = torch.where(torch.logical_and(pred_img == float(0.), pred_MVS_img != float(0.)),
                                   pred_MVS_img,
                                   pred_img)
            conf_img = torch.where(torch.logical_and(conf_img == float(0.), conf_MVS_img != float(0.)),
                                   conf_MVS_img,
                                   conf_img)

            return pred_img, conf_img
        else:
            raise ValueError(f'depth type should be in original, mesh or merge but got: {depth_type}')

    @staticmethod
    def int_to_viewdir(i: int):
        return f"view_{i:05d}"
    
    def load_face_bounds(self, scan_path):
        if not (scan_path / "face_vertices.npy").exists():
            raise FileNotFoundError("Couldnt find vertices file for scan", scan_path)
        vertices_path = scan_path / "face_vertices.npy"
        with open(vertices_path, 'r') as vertices_file:
            vertices = [list(map(float, line.split())) for line in vertices_file]
        # Convert the list of lists to a NumPy array
        xyz = np.array(vertices).astype(np.float32)
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[2] -= 0.05
        max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        return bounds

    @staticmethod
    def get_mask_at_box(bounds, K, R, T, H, W):
        ray_o, ray_d = FacescapeDataSet.get_rays(H, W, K, R, T)

        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = FacescapeDataSet.get_near_far(bounds, ray_o, ray_d)
        return mask_at_box.reshape((H, W))

    @staticmethod
    def get_rays(H, W, K, R, T):
        rays_o = -np.dot(R.T, T).ravel()

        i, j = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32), indexing='xy')

        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_world = np.dot(pixel_camera - T.ravel(), R)
        rays_d = pixel_world - rays_o[None, None]
        rays_o = np.broadcast_to(rays_o, rays_d.shape)

        return rays_o, rays_d
    
    @staticmethod
    def get_near_far(bounds, ray_o, ray_d, boffset=(-0.01, 0.01)):
        """calculate intersections with 3d bounding box"""
        bounds = bounds + np.array([boffset[0], boffset[1]])[:, None]
        nominator = bounds[None] - ray_o[:, None]
        # calculate the step of intersections at six planes of the 3d bounding box
        ray_d[np.abs(ray_d) < 1e-5] = 1e-5
        d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
        # calculate the six interections
        p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
        # calculate the intersections located at the 3d bounding box
        min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
        eps = 1e-6
        p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                        (p_intersect[..., 0] <= (max_x + eps)) * \
                        (p_intersect[..., 1] >= (min_y - eps)) * \
                        (p_intersect[..., 1] <= (max_y + eps)) * \
                        (p_intersect[..., 2] >= (min_z - eps)) * \
                        (p_intersect[..., 2] <= (max_z + eps))
        # obtain the intersections of rays which intersect exactly twice
        mask_at_box = p_mask_at_box.sum(-1) == 2
        p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
            -1, 2, 3)

        # calculate the step of intersections
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        norm_ray = np.linalg.norm(ray_d, axis=1)
        d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
        d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
        near = np.minimum(d0, d1)
        far = np.maximum(d0, d1)

        return near, far, mask_at_box

    def get_metas(self):
        meta_dir = Path("assets/data_splits/facescape")
        meta_fpath = meta_dir / (self.stage + "_metas_binocular.txt")
        if meta_fpath.exists():
            with open(meta_fpath, "r") as f:
                metas = json.load(f)
        else:
            raise ValueError('Get Metas not implemented. Look at $HOME/metas.py file for reference.')
        # Repeat metas 20 times for 150000 training samples and 1600 val samples
        # Repeat metas 5 times for 37000 training samples and 400 val samples
        n_repeat = 5 if self.stage == 'train' else 20
        repeat_metas = list(itertools.chain.from_iterable(itertools.repeat(meta, n_repeat) for meta in metas))
        return repeat_metas

    def __len__(self):
        return len(self.metas)

    @staticmethod
    def get_frame_n_subject(scan_path):
        frame, subject = scan_path.name, scan_path.parent.name
        return frame, subject

    def __getitem__(self, idx):
        while True:
            if True:
                meta = self.metas[idx]
                
                suffix = "_val" if self.stage == 'val' else ""
                target_ids = np.array(meta["targets" + suffix])
                left_ids = np.array(meta["l_refs" + suffix])
                right_ids = np.array(meta["r_refs" + suffix])
                target_id = self.rnd.choice(target_ids)
                left_id = self.rnd.choice(left_ids)
                right_id = self.rnd.choice(right_ids)
                source_ids = [left_id, right_id]

                scan_path = self.data_dir / meta["scan_path"]
                meta_path = Path(meta["scan_path"])
                sample_path = scan_path / self.int_to_viewdir(int(target_id))
                source_paths = [scan_path / self.int_to_viewdir(int(source_id)) for source_id in source_ids]
                source_depth_paths = [meta_path / self.int_to_viewdir(int(source_id)) for source_id in source_ids]
                cam_path = scan_path / "cameras.json"

                frame, subject = self.get_frame_n_subject(scan_path)

                target_rgba_path = sample_path / self.RGBA_FNAME
                src_rgba_paths = [source_path / self.RGBA_FNAME for source_path in source_paths]
                
                src_depth_paths = [source_path / self.DEPTH_FNAME for source_path in source_depth_paths]
                src_depth_paths = [os.path.join(
                    "/cluster/home/tguillou/depths_gt_pred_conf",
                    '_'.join(str(src_depth_path).split('/'))
                ) for src_depth_path in src_depth_paths]
                
                src_mesh_depth_paths = [source_path / self.DEPTH_MESH_FNAME for source_path in source_depth_paths]
                src_mesh_depth_paths = [os.path.join(
                    "/cluster/home/tguillou/depths_mesh",
                    '_'.join(str(src_depth_path).split('/'))
                ) for src_depth_path in src_mesh_depth_paths]

                target_rgb, target_alpha = self.read_rgba(target_rgba_path)
                image_shape = target_rgb.shape[1:]
                if self.model in ['DINER', 'OURS']:
                    src_rgbs = list()
                    src_alphas = list()
                    src_depths = list()
                    src_depth_stds = list()
                    for src_rgba_path, src_depth_path, src_mesh_depth_path in \
                            zip(src_rgba_paths, src_depth_paths, src_mesh_depth_paths):
                        src_rgb, src_alpha = self.read_rgba(src_rgba_path)
                        src_depth, src_depth_std = self.read_depth(src_depth_path, src_mesh_depth_path, depth_type=self.depth_type)
                        # src_depth = self.read_depth(src_depth_path)
                        # src_depth_std = self.read_depth(src_depth_std_path)
                        src_rgbs.append(src_rgb), src_alphas.append(src_alpha), src_depths.append(src_depth)
                        src_depth_stds.append(src_depth_std)

                    src_rgbs = torch.stack(src_rgbs)
                    src_depths = torch.stack(src_depths)
                    src_depth_stds = torch.stack(src_depth_stds)
                    src_depth_stds = self.conf2std(src_depth_stds)
                    src_alphas = torch.stack(src_alphas)

                    with open(cam_path, "r") as f:
                        cam_dict = json.load(f)
                    target_extrinsics = torch.tensor(cam_dict[target_id]["extrinsics"])
                    src_extrinsics = torch.tensor([cam_dict[src_id]["extrinsics"] for src_id in source_ids])
                    target_extrinsics = to_homogeneous_trafo(target_extrinsics[None])[0]
                    src_extrinsics = to_homogeneous_trafo(src_extrinsics)
                    target_intrinsics = torch.tensor(cam_dict[target_id]["intrinsics"])
                    src_intrinsics = torch.tensor([cam_dict[src_id]["intrinsics"] for src_id in source_ids])

                    sample = dict(target_rgb=target_rgb,
                                  target_alpha=target_alpha,
                                  target_extrinsics=target_extrinsics,
                                  target_intrinsics=target_intrinsics,
                                  target_view_id=torch.tensor(int(target_id)),
                                  scan_idx=0,
                                  sample_name=f"{subject}-{frame}-{target_id}-{'-'.join(source_ids)}-",
                                  frame=frame,
                                  src_rgbs=src_rgbs,
                                  src_depths=src_depths,
                                  src_depth_stds=src_depth_stds,
                                  src_alphas=src_alphas,
                                  src_extrinsics=src_extrinsics,
                                  src_intrinsics=src_intrinsics,
                                  src_view_ids=torch.tensor([int(src_id) for src_id in source_ids]))

                    sample = dict_2_torchdict(sample)

                    return sample
                else:
                    abs_kpt3d_path = scan_path / "3dlmks.npy"
                    with open(abs_kpt3d_path, 'r') as kpt3d_file:
                        abs_target_kpt3d = [list(map(float, line.split())) for line in kpt3d_file]
                    # Convert the list of lists to a NumPy array
                    abs_target_kpt3d = np.array(abs_target_kpt3d).astype(np.float32)
                    abs_target_kpt3d=torch.from_numpy(abs_target_kpt3d)
                    
                    src_rgbs = list()
                    src_alphas = list()
                    src_masks = list()
                    for src_rgba_path in src_rgba_paths:
                        src_rgb, src_alpha = self.read_rgba(src_rgba_path)
                        src_mask = src_rgb.sum(0) != 3
                        src_rgb[~(src_mask.repeat(3, 1, 1))] = 0
                        src_rgbs.append(src_rgb), src_alphas.append(src_alpha), src_masks.append(src_mask)

                    src_rgbs = torch.stack(src_rgbs)
                    src_alphas = torch.stack(src_alphas)
                    src_masks = torch.stack(src_masks)
                    
                    with open(cam_path, "r") as f:
                        cam_dict = json.load(f)
                    target_extrinsics = torch.tensor(cam_dict[target_id]["extrinsics"])
                    src_extrinsics = torch.tensor([cam_dict[src_id]["extrinsics"] for src_id in source_ids])
                    target_extrinsics = to_homogeneous_trafo(target_extrinsics[None])[0] # (4, 4)
                    src_extrinsics = to_homogeneous_trafo(src_extrinsics)  # (2, 4, 4)
                    target_intrinsics = torch.tensor(cam_dict[target_id]["intrinsics"])  # (3, 3)
                    src_intrinsics = torch.tensor([cam_dict[src_id]["intrinsics"] for src_id in source_ids]) # (2, 3, 3)
                    
                    target_mask = (target_rgb.sum(0) != 3)
                    target_rgb[~(target_mask.repeat(3, 1, 1))] = 0
                    
                    target_kpt3d = abs_target_kpt3d
                    
                    bounds = self.load_face_bounds(scan_path)
                    H, W = image_shape
                    mask_at_box = self.get_mask_at_box(
                        bounds,
                        target_intrinsics.numpy(),
                        target_extrinsics[:3, :3].numpy(),
                        target_extrinsics[:3, -1].numpy(),
                        H, W)
                    mask_at_box = mask_at_box.reshape((H, W))

                    sample = dict(target_rgb=target_rgb,
                                  target_alpha=target_alpha,
                                  target_extrinsics=target_extrinsics,
                                  target_intrinsics=target_intrinsics,
                                  target_kpt3d=target_kpt3d,
                                  target_mask=target_mask,
                                  target_view_id=torch.tensor(int(target_id)),
                                  scan_idx=0,
                                  bounds=bounds,
                                  mask_at_box=mask_at_box,
                                  sample_name=f"{subject}-{frame}-{target_id}-{'-'.join(source_ids)}-",
                                  frame=frame,
                                  src_rgbs=src_rgbs,
                                  src_alphas=src_alphas,
                                  src_extrinsics=src_extrinsics,
                                  src_intrinsics=src_intrinsics,
                                  src_mask=src_masks,
                                  src_view_ids=torch.tensor([int(src_id) for src_id in source_ids]))

                    sample = dict_2_torchdict(sample)

                    return sample

    def get_cam_sweep_extrinsics(self, nframes, scan_idx, elevation=0., radius=1.8, sweep_range=None):
        base_sample = self.__getitem__(scan_idx)
        device = base_sample["target_extrinsics"].device

        src_extrinsics = base_sample["src_extrinsics"]
        src_centers = -1 * src_extrinsics[:, :3, :3].permute(0, 2, 1) @ src_extrinsics[:, :3, -1:]  # N x 3 x 1
        src_dirs = src_centers[..., 0] / torch.norm(src_centers[..., 0], p=2, keepdim=True, dim=-1)  # N x 3
        mean_dir = src_dirs.sum(dim=0)
        mean_dir /= torch.norm(mean_dir, p=2, dim=0)
        center = mean_dir * radius
        z_ax = -center / torch.norm(center, p=2)
        y_ax = torch.tensor([0., 0., -1], device=device, dtype=torch.float)
        x_ax = torch.cross(y_ax, z_ax)
        x_ax /= torch.norm(x_ax, p=2)

        base_pose = torch.eye(4, device=device, dtype=torch.float)
        base_pose[:3, 0] = x_ax
        base_pose[:3, 1] = y_ax
        base_pose[:3, 2] = z_ax
        base_pose[:3, 3] = center

        sweep_range = sweep_range if sweep_range is not None else self.range_hor
        # base_sample_idx = scan_idx * self.ncams
        # base_sample = self.__getitem__(base_sample_idx)

        rotations = [torch.tensor([[np.cos(alpha), -np.sin(alpha), 0, 0],
                                   [np.sin(alpha), np.cos(alpha), 0, 0],
                                   [0., 0., 1, 0],
                                   [0., 0., 0., 1.]], device=device, dtype=torch.float)
                     for alpha in np.linspace(-sweep_range / 180 * np.pi, sweep_range / 180 * np.pi, nframes)]
                     # for alpha in np.array([70., 48, 25, 2, -21,]) / 180 * np.pi]  # enable for teaser
        rotations = torch.stack(rotations)

        target_poses = rotations @ base_pose[None].expand(nframes, -1, -1)
        target_extrinsics = torch.linalg.inv(target_poses)

        # # visualize cam sweep poses
        # import matplotlib.pyplot as plt
        # target_poses = torch.linalg.inv(target_extrinsics)
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # s = .1
        # for i, color in enumerate(["red", "green", "blue"]):
        #     ax.quiver(target_poses[:, 0, -1], target_poses[:, 1, -1], target_poses[:, 2, -1],
        #               s * target_poses[:, 0, i], s * target_poses[:, 1, i], s * target_poses[:, 2, i],
        #               edgecolor=color)
        # for i, id in enumerate(range(nframes)):
        #     ax.text(target_poses[i, 0, -1], target_poses[i, 1, -1], target_poses[i, 2, -1], str(id))
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # ax.set_xlim(-1.5, 1.5)
        # ax.set_ylim(-1.5, 1.5)
        # ax.set_zlim(-1.5, 1.5)
        # plt.show()
        # plt.close()

        return target_extrinsics

    def visualize_item(self, idx):
        """
        plots item for debugging purposes
        :param idx:
        :return:
        """
        sample = self.__getitem__(idx)

        print(sample["target_view_id"], sample["src_view_ids"])

        # visualizing target and source views (rgb, alpha)
        import matplotlib.pyplot as plt
        ncols = self.nsource + 1
        nrows = 3
        s = 3
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(s * ncols, s * nrows))
        axes[0, 0].imshow(sample["target_rgb"].permute(1, 2, 0))
        axes[1, 0].imshow(sample["target_alpha"].permute(1, 2, 0))
        axes[0, 0].set_title(str(sample["target_view_id"]))
        for i in range(self.nsource):
            depth_masked = sample["src_depths"][i][sample["src_depths"][i] != 0]
            axes[0, i + 1].imshow(sample["src_rgbs"][i].permute(1, 2, 0))
            axes[1, i + 1].imshow(sample["src_alphas"][i].permute(1, 2, 0))
            axes[0, i + 1].set_title(str(sample["src_view_ids"][i]))
            axes[2, i + 1].imshow(sample["src_depths"][i].permute(1, 2, 0), vmin=depth_masked.min(),
                                  vmax=depth_masked.max())
        [a.axis("off") for a in axes.flatten()]
        fig.suptitle(sample["sample_name"])
        plt.show()
        plt.close()

        # visualizing camera positions
        import matplotlib.pyplot as plt
        targ_extrinsics = sample["target_extrinsics"]
        src_extrinsics = sample["src_extrinsics"]
        all_extrinsics = torch.cat((targ_extrinsics.unsqueeze(0), src_extrinsics), dim=0)
        all_centers = -all_extrinsics[:, :3, :3].permute(0, 2, 1) @ all_extrinsics[:, :3, -1:]
        all_ids = [sample["target_view_id"].item()] + sample["src_view_ids"].tolist()

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        s = .1
        for i, color in enumerate(["red", "green", "blue"]):
            ax.quiver(all_centers[:, 0, 0], all_centers[:, 1, 0], all_centers[:, 2, 0],
                      s * all_extrinsics[:, i, 0], s * all_extrinsics[:, i, 1], s * all_extrinsics[:, i, 2],
                      edgecolor=color)
        for i, id in enumerate(all_ids):
            ax.text(all_centers[i, 0, 0], all_centers[i, 1, 0], all_centers[i, 2, 0], str(id))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((-1.5, 1.5))
        ax.set_zlim((-1.5, 1.5))
        plt.show()
        plt.close()

    def visualize_camgrid(self, i=0):
        """
        plots item for debugging purposes
        :param idx:
        :return:
        """

        # visualizing camera positions
        import matplotlib.pyplot as plt
        meta = self.metas[i]
        print(meta)
        cam_path = self.data_dir / meta["scan_path"] / "cameras.json"
        with open(cam_path, "r") as f:
            cam_dict = json.load(f)
        all_ids = sorted(cam_dict.keys())
        all_extrinsics = torch.tensor([cam_dict[i]["extrinsics"] for i in all_ids])

        all_centers = -all_extrinsics[:, :3, :3].permute(0, 2, 1) @ all_extrinsics[:, :3, -1:]

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        s = .1
        for i, color in enumerate(["red", "green", "blue"]):
            ax.quiver(all_centers[:, 0, 0], all_centers[:, 1, 0], all_centers[:, 2, 0],
                      s * all_extrinsics[:, i, 0], s * all_extrinsics[:, i, 1], s * all_extrinsics[:, i, 2],
                      edgecolor=color)
        for i, id in enumerate(all_ids):
            ax.text(all_centers[i, 0, 0], all_centers[i, 1, 0], all_centers[i, 2, 0], id)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
        plt.close()

    def reproject_depth(self, sample_idx=0, outfile=None):
        """
        creates point cloud from depth maps of sample and optionally saves it to outfile
        :param sample_idx:
        :param outfile:
        :return:
        """
        sample = self.__getitem__(sample_idx)
        src_imgs = sample["src_rgbs"]
        src_depths = sample["src_depths"]
        K = sample["src_intrinsics"]
        Rt = sample["src_extrinsics"]
        K_inv = torch.linalg.inv(K)
        Rt_inv = torch.linalg.inv(Rt)
        N = len(src_depths)

        # create image rays
        H, W = src_imgs.shape[-2:]
        src_rays = torch.stack(torch.meshgrid(torch.arange(0.5, H, step=1.), torch.arange(0.5, W, step=1.))[::-1],
                               dim=-1)  # (H, W, 2)
        src_rays = torch.cat((src_rays, torch.ones_like(src_rays[..., :1])), dim=-1)  # (H, W, 3)
        src_rays = src_rays[None].expand(N, -1, -1, -1)
        src_rays = (K_inv @ src_rays.reshape(N, -1, 3).permute(0, 2, 1)).permute(0, 2, 1)  # (N, H*W, 3)

        # projection into world space
        src_points = src_rays * src_depths[:, 0].reshape(N, H * W)[..., None]  # (N, H * W, 3)
        src_points = torch.cat((src_points, torch.ones_like(src_points[..., :1])), dim=-1)  # (N, H * W, 4)
        world_points = (Rt_inv @ src_points.permute(0, 2, 1)).permute(0, 2, 1)  # (N, H * W, 4)

        world_points = world_points[..., :3].reshape(-1, 3)  # (N*H*W, 3)
        colors = src_imgs.permute(0, 2, 3, 1).reshape(-1, 3)  # (N*H*W, 3)

        if outfile is not None:
            out = torch.cat((world_points, (colors * 255).round()), dim=-1).cpu().numpy()
            np.savetxt(outfile, out, delimiter=";")

        return world_points, colors

    def check_depth_existence(self):
        missing_depths = []
        depth_paths_old = ""
        for meta in tqdm.tqdm(self.metas, desc="Checking Depth Images"):
            scan_path = self.data_dir / meta["scan_path"]
            source_ids = [s_ids[0] for s_ids in meta["ref_ids"]]
            source_ids = np.unique(np.array(source_ids).flatten())
            depth_paths = [scan_path / self.int_to_viewdir(int(id)) / self.DEPTH_FNAME for id in source_ids]
            depth_paths_new = ",".join([str(dp) for dp in depth_paths])
            if depth_paths_new == depth_paths_old:
                pass
            else:
                depth_paths_old = depth_paths_new
                for depth_path in depth_paths:
                    if not depth_path.exists():
                        missing_depths.append(depth_path)
        if missing_depths:
            raise FileNotFoundError("Missing depth files", missing_depths)
