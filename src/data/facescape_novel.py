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
import cv2
from src.util.cam_geometry import to_homogeneous_trafo, project_to_relative_coordinates

OPENCV2OPENGL = np.array([[1., 0., 0., 0.], [0., -1., 0., 0], [0., 0., -1., 0.], [0., 0., 0., 1.]], dtype=np.float32)


class FacescapeDataSet(torch.utils.data.Dataset):
    znear = 1.
    zfar = 2.5
    RGBA_FNAME = "rgba_colorcalib_v2.png"
    DEPTH_FNAME = "depth_gt_pred_conf.png"
    DEPTH_MESH_FNAME = "depth_mesh.png"

    def __init__(self, model, root: Path, stage, range_hor=45, range_vert=30, slide_range=40, slide_step=20, depth_fname=None, data_type=None):
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
        self.data_type = data_type
        closed_eyes = ["18"]
        open_mouth = ["03", "13", "16"]

        self.unwanted_ref_expr = list()
        self.unwanted_tgt_expr = list()
        if data_type == "NOO":
            self.unwanted_ref_expr = closed_eyes
            self.unwanted_tgt_expr = closed_eyes + open_mouth
        elif data_type == "NCO":
            self.unwanted_ref_expr = closed_eyes
            self.unwanted_tgt_expr = open_mouth
        elif data_type == "NOC":
            self.unwanted_ref_expr = closed_eyes + open_mouth
            self.unwanted_tgt_expr = closed_eyes + open_mouth
        elif data_type == "NCC":
            self.unwanted_ref_expr = closed_eyes + open_mouth
            self.unwanted_tgt_expr = open_mouth
        
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
        self.conf2std = self._getconf2std()
        self.metas = self.get_metas()
        self.gen_vertices, self.gen_extrinsics, self.gen_intrinsics = self.get_general()
        
    def get_general(self):
        gen_path = self.data_dir / "003/03"
        gen_vertices_path = gen_path  / "face_vertices.npy"
        with open(gen_vertices_path, 'r') as gen_vertices_file:
            gen_vertices = [list(map(float, line.split())) for line in gen_vertices_file]
        # Convert the list of lists to a NumPy array
        gen_vertices = np.array(gen_vertices).astype(np.float32)
        gen_vertices = torch.from_numpy(gen_vertices)
        
        gen_cam_path = gen_path  / "cameras.json"
        with open(gen_cam_path, "r") as f:
            gen_cam_dict = json.load(f)
        gen_intrinsics = torch.tensor(gen_cam_dict["18"]["intrinsics"])
        gen_extrinsics = torch.tensor(gen_cam_dict["18"]["extrinsics"])
        
        return gen_vertices, gen_extrinsics, gen_intrinsics

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
    def read_depth(mesh_path: Path, transmvsnet_path: Path = None):
        UINT16_MAX = 65535
        SCALE_FACTOR = 1e-4
        
        unprocessed_mesh_depth_img = Image.open(mesh_path)
        
        pred_img = pil_to_tensor(unprocessed_mesh_depth_img).float() * SCALE_FACTOR
        conf_img = torch.where(pred_img == float(0.),
                               float(0.),
                               float(0.8))
        
        if transmvsnet_path is not None:
            unprocessed_depth_img = Image.open(transmvsnet_path)
            
            width = unprocessed_depth_img.width // 3
            gt_pil = unprocessed_depth_img.crop((0, 0, width, unprocessed_depth_img.height))
            pred_pil = unprocessed_depth_img.crop((width, 0, 2 * width, unprocessed_depth_img.height))
            conf_pil = unprocessed_depth_img.crop((2 * width, 0,
                                                   unprocessed_depth_img.width,
                                                   unprocessed_depth_img.height))
            pred_MVS_img = pil_to_tensor(pred_pil).float() * SCALE_FACTOR
            conf_MVS_img = pil_to_tensor(conf_pil).float() * SCALE_FACTOR

            # Final Step, Union of both images
            pred_img = torch.where(torch.logical_and(pred_img == float(0.), pred_MVS_img != float(0.)),
                                   pred_MVS_img,
                                   pred_img)
            conf_img = torch.where(torch.logical_and(conf_img == float(0.), conf_MVS_img != float(0.)),
                                   conf_MVS_img,
                                   conf_img)
        
        return pred_img, conf_img

    @staticmethod
    def int_to_viewdir(i: int):
        return f"view_{i:05d}"

    def get_metas(self):
        meta_dir = Path("assets/data_splits/facescape")
        meta_fpath = meta_dir / (self.stage + "_novel_pose_metas.txt")
        if meta_fpath.exists():
            with open(meta_fpath, "r") as f:
                metas = json.load(f)
        else:
            raise ValueError('Get Metas not implemented. Look at $HOME/metas.py file for reference.')
        # Repeat metas 400 times for 150000 training samples and 1600 val samples
        # Repeat metas 100 times for 37000 training samples and 400 val samples
        new_metas = list()
        new_metas = [meta for meta in metas if meta["subject"] not in ["095", "160", "170", "291"]]
        
        n_repeat = 100 if self.stage == 'train' else 400
        repeat_metas = list(itertools.chain.from_iterable(itertools.repeat(meta, n_repeat) for meta in new_metas))
        return repeat_metas

    def __len__(self):
        return len(self.metas)

    @staticmethod
    def get_frame_n_subject(scan_path):
        frame, subject = scan_path.name, scan_path.parent.name
        return frame, subject

    def __getitem__(self, idx):
        while True:
            meta = self.metas[idx]

            # Step 1: Get a ref expression and a target expression
            while True:
                ref_expression = self.rnd.choice(meta["ref_expressions"])
                if ref_expression["expression"] not in self.unwanted_ref_expr:
                    break
                
            while True:
                target_expression = self.rnd.choice(meta["target_expressions"])
                if target_expression["expression"] not in self.unwanted_tgt_expr:
                    break

            # Step 2: Get the ref ids and target ids
            target_ids = np.array(target_expression["targets"])
            left_ids = np.array(ref_expression["left_refs"])
            right_ids = np.array(ref_expression["right_refs"])
            
            target_id = self.rnd.choice(target_ids)
            left_id = self.rnd.choice(left_ids)
            right_id = self.rnd.choice(right_ids)
            source_ids = [left_id, right_id]

            # Step 3: Get ref scan path and target scan path
            ref_scan_path = self.data_dir / meta["subject"] / ref_expression["expression"]
            target_scan_path = self.data_dir / meta["subject"] / target_expression["expression"]

            meta_path = Path(meta["subject"]) / ref_expression["expression"]
            sample_path = target_scan_path / self.int_to_viewdir(int(target_id))
            source_paths = [ref_scan_path / self.int_to_viewdir(int(source_id)) for source_id in source_ids]
            source_depth_paths = [meta_path / self.int_to_viewdir(int(source_id)) for source_id in source_ids]
            ref_cam_path = ref_scan_path / "cameras.json"
            target_cam_path = target_scan_path / "cameras.json"

            ref_frame, subject = self.get_frame_n_subject(ref_scan_path)
            target_frame, _ = self.get_frame_n_subject(target_scan_path)

            target_rgba_path = sample_path / self.RGBA_FNAME
            src_rgba_paths = [source_path / self.RGBA_FNAME for source_path in source_paths]

            src_depth_paths = [source_path / self.DEPTH_FNAME for source_path in source_depth_paths]
            src_depth_paths = [os.path.join(
                "/cluster/home/tguillou/depths_gt_pred_conf",
                '_'.join(str(src_depth_path).split('/'))
            ) for src_depth_path in src_depth_paths]

            src_mesh_depth_paths = [source_path / self.DEPTH_MESH_FNAME for source_path in source_depth_paths]
            src_mesh_depth_paths = [os.path.join(
                "/cluster/home/tguillou/novel_depths_mesh",
                '_'.join(str(src_depth_path).split('/'))
            ) for src_depth_path in src_mesh_depth_paths]
            target_mesh_depth_path = os.path.join(
                "/cluster/home/tguillou/target_depths_mesh",
                '_'.join(str(sample_path).split('/')[-3:] + [self.DEPTH_MESH_FNAME])
            )
            
            ref_vertices_path = ref_scan_path / "face_vertices.npy"
            with open(ref_vertices_path, 'r') as ref_vertices_file:
                ref_vertices = [list(map(float, line.split())) for line in ref_vertices_file]
            # Convert the list of lists to a NumPy array
            ref_vertices = np.array(ref_vertices).astype(np.float32)
            ref_vertices = torch.from_numpy(ref_vertices)

            target_vertices_path = target_scan_path / "face_vertices.npy"
            with open(target_vertices_path, 'r') as target_vertices_file:
                target_vertices = [list(map(float, line.split())) for line in target_vertices_file]
            # Convert the list of lists to a NumPy array
            target_vertices = np.array(target_vertices).astype(np.float32)
            target_vertices = torch.from_numpy(target_vertices)

            offset_target_to_source = ref_vertices - target_vertices
                
            target_rgb, target_alpha = self.read_rgba(target_rgba_path)
            # target_depth, target_depth_std = self.read_depth(target_mesh_depth_path)
            image_shape = target_rgb.shape[1:]

            src_rgbs = list()
            src_alphas = list()
            src_depths = list()
            src_depth_stds = list()
            for src_rgba_path, src_depth_path, src_mesh_depth_path in \
                    zip(src_rgba_paths, src_depth_paths, src_mesh_depth_paths):
                src_rgb, src_alpha = self.read_rgba(src_rgba_path)
                src_depth, src_depth_std = self.read_depth(src_mesh_depth_path) # , transmvsnet_path=src_depth_path)
                src_rgbs.append(src_rgb), src_alphas.append(src_alpha), src_depths.append(src_depth)
                src_depth_stds.append(src_depth_std)

            src_rgbs = torch.stack(src_rgbs)
            src_depths = torch.stack(src_depths)
            src_depth_stds = torch.stack(src_depth_stds)
            src_depth_stds = self.conf2std(src_depth_stds)
            src_alphas = torch.stack(src_alphas)

            with open(ref_cam_path, "r") as f:
                ref_cam_dict = json.load(f)
            with open(target_cam_path, "r") as f:
                target_cam_dict = json.load(f)
            target_intrinsics = torch.tensor(target_cam_dict[target_id]["intrinsics"])
            src_intrinsics = torch.tensor([ref_cam_dict[src_id]["intrinsics"] for src_id in source_ids])
            target_extrinsics = torch.tensor(target_cam_dict[target_id]["extrinsics"])
            src_extrinsics = torch.tensor([ref_cam_dict[src_id]["extrinsics"] for src_id in source_ids])
            
            target_extrinsics = to_homogeneous_trafo(target_extrinsics[None])[0]
            src_extrinsics = to_homogeneous_trafo(src_extrinsics)

            sample = dict(target_rgb=target_rgb,
                          # target_depth=target_depth,
                          # target_depth_std=target_depth_std,
                          target_alpha=target_alpha,
                          target_extrinsics=target_extrinsics,
                          target_intrinsics=target_intrinsics,
                          target_vertices=target_vertices,
                          target_view_id=torch.tensor(int(target_id)),
                          scan_idx=0,
                          sample_name=f"{subject}-{ref_frame}-{target_frame}-{target_id}-{'-'.join(source_ids)}-",
                          ref_frame=ref_frame,
                          target_frame=target_frame,
                          src_rgbs=src_rgbs,
                          src_depths=src_depths,
                          src_depth_stds=src_depth_stds,
                          src_alphas=src_alphas,
                          src_extrinsics=src_extrinsics,
                          src_intrinsics=src_intrinsics,
                          src_vertices=ref_vertices,
                          src_view_ids=torch.tensor([int(src_id) for src_id in source_ids]),
                          offset_target_to_source=offset_target_to_source,
                          gen_extrinsics = self.gen_extrinsics,
                          gen_intrinsics = self.gen_intrinsics,
                          offset_target_to_gen = self.gen_vertices - target_vertices)

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
