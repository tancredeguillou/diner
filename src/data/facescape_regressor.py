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
    RGBA_FNAME = "rgba_colorcalib_v2.png"

    def __init__(self, model, root: Path, stage, data_type=None):
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
        self.metas = self.get_metas()

    @staticmethod
    def read_rgba(p: Path, symmetric_range=False, bg=1.):
        rgb, a = torch.split(pil_to_tensor(Image.open(p)).float() / 255., (3, 1))  # range: 0 ... 1

        if symmetric_range:  # rgb range: -1 ... 1
            rgb = rgb * 2 - 1

        rgb.permute(1, 2, 0)[a[0] < .5] = bg
        return rgb, a

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

            # Step 1: Get a ref expression
            while True:
                ref_expression = self.rnd.choice(meta["ref_expressions"])
                if ref_expression["expression"] not in self.unwanted_ref_expr:
                    break

            # Step 2: Get the image id
            left_ids = np.array(ref_expression["left_refs"])
            right_ids = np.array(ref_expression["right_refs"])
            left_id = self.rnd.choice(left_ids)
            right_id = self.rnd.choice(right_ids)
            source_ids = [left_id, right_id]
            
            img_id = self.rnd.choice(source_ids)

            # Step 3: Get ref scan path and target scan path
            img_scan_path = self.data_dir / meta["subject"] / ref_expression["expression"]
            img_path = img_scan_path / self.int_to_viewdir(int(img_id))
            img_cam_path = img_scan_path / "cameras.json"

            img_frame, subject = self.get_frame_n_subject(img_scan_path)

            img_rgba_path = img_path / self.RGBA_FNAME

            img_vertices_path = img_scan_path / "face_vertices.npy"
            with open(img_vertices_path, 'r') as img_vertices_file:
                img_vertices = [list(map(float, line.split())) for line in img_vertices_file]
            # Convert the list of lists to a NumPy array
            img_vertices = np.array(img_vertices).astype(np.float32)
            img_vertices = torch.from_numpy(img_vertices) # (NP, 3)
            
            with open(img_cam_path, "r") as f:
                img_cam_dict = json.load(f)
            img_intrinsics = torch.tensor(img_cam_dict[img_id]["intrinsics"]) # (3, 3)
            img_extrinsics = torch.tensor(img_cam_dict[img_id]["extrinsics"]) # (3, 4)
                
            image, _ = self.read_rgba(img_rgba_path)
            
            target_keypoints = project_to_relative_coordinates(img_vertices, img_extrinsics, img_intrinsics) # (NP, 2) - 5.2, 278.3

            sample = dict(image=image,
                          target_keypoints=target_keypoints,
                          sample_name=f"{subject}-{img_frame}-{img_id}",
                          img_frame=img_frame,
                          img_view_id=torch.tensor(int(img_id)))
            sample = dict_2_torchdict(sample)

            return sample