import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Slerp as RotSlerp


def project_3d_to_2d(kpt3d, extrinsics, intrinsics):
    """
    Project 3D keypoints to 2D image plane.

    Parameters:
    - kpt3d: 3D keypoints (absolute coordinates) [N, 3].
    - extrinsics: Extrinsic matrix for the camera [B, 4, 4].
    - intrinsics: Intrinsic matrix for the camera [B, 3, 3].

    Returns:
    - kpt2d: 2D keypoints (projected coordinates) [B, N, 2].
    """
    n_views = extrinsics.shape[0]
    kpt3d_h = torch.cat((kpt3d, torch.ones_like(kpt3d[:, :1])), dim=1)
    intrin = torch.eye(4, device=intrinsics.device)[None].repeat(n_views, 1, 1)
    intrin[:, :3, :3] = intrinsics.clone()
    kpt2d_homogeneous = torch.matmul(torch.bmm(extrinsics, intrin), kpt3d_h.unsqueeze(0).permute(0, 2, 1))
    kpt2d = kpt2d_homogeneous[:, :2, :] / kpt2d_homogeneous[:, 2:, :]
    return kpt2d.permute(0, 2, 1)


def transform_absolute_to_camera_coordinates(absolute_coordinates, extrinsics):
    """
    Transform 3D absolute coordinates to 3D camera coordinates.

    Parameters:
    - absolute_coordinates: Tensor of shape (N, 3) representing 3D absolute coordinates.
    - extrinsics: Tensor of shape (4, 4) representing camera extrinsics matrix.

    Returns:
    - camera_coordinates: Tensor of shape (N, 3) representing 3D coordinates in camera space.
    """
    # Homogeneous coordinates (adding a column of ones)
    absolute_coordinates_homogeneous = torch.cat([absolute_coordinates, torch.ones(absolute_coordinates.shape[0], 1)], dim=1)

    # Apply extrinsics transformation
    camera_coordinates_homogeneous = torch.matmul(absolute_coordinates_homogeneous, extrinsics.T)

    # Extract camera coordinates (remove homogeneous component)
    camera_coordinates = camera_coordinates_homogeneous[:, :3]

    return camera_coordinates


def transform_absolute_to_camera_coordinates_np(absolute_coordinates, extrinsics):
    """
    Transform 3D absolute coordinates to 3D camera coordinates.

    Parameters:
    - absolute_coordinates: NumPy array of shape (N, 3) representing 3D absolute coordinates.
    - extrinsics: NumPy array of shape (4, 4) representing camera extrinsics matrix.

    Returns:
    - camera_coordinates: NumPy array of shape (N, 3) representing 3D coordinates in camera space.
    """
    # Homogeneous coordinates (adding a column of ones)
    absolute_coordinates_homogeneous = np.column_stack((absolute_coordinates, np.ones(absolute_coordinates.shape[0])))

    # Apply extrinsics transformation
    camera_coordinates_homogeneous = np.dot(absolute_coordinates_homogeneous, extrinsics.T)

    # Extract camera coordinates (remove homogeneous component)
    camera_coordinates = camera_coordinates_homogeneous[:, :3]

    return camera_coordinates


def compute_mask_radius(center_kpt2D, kpts2d):
    """
    Compute the mask radius as the maximum distance between the center and any keypoint.

    Parameters:
    - center_kpt2D: 2D center.
    - target_kpt3d: 2D keypoints [N, 2].

    Returns:
    - mask_radius: Computed mask radius.
    """
    # Calculate the distances from the center to all keypoints
    distances = torch.norm(kpts2d - center_kpt2D, dim=1)

    # Use the maximum distance as the mask radius
    mask_radius = torch.max(distances)

    return mask_radius.item()


def generate_circular_mask(image_shape, center, radius):
    """
    Generate a circular mask.

    Parameters:
    - image_shape: Shape of the image [H, W].
    - center: Center coordinates of the circle [B, 2].
    - radius: Radius of the circle.

    Returns:
    - mask: Binary circular mask [B, H, W].
    """
    h, w = image_shape
    Y, X = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    mask = ((Y - center[:, 1].view(-1, 1, 1))**2 + (X - center[:, 0].view(-1, 1, 1))**2 <= radius**2).float()
    return mask


def gen_rays(extrinsics, intrinsics, W, H, z_near, z_far):
    """
    Calculates camera rays.
    Parameters
    ----------
    extrinsics: (B, 4, 4)
    width: int
    height: int
    focal: (B,2) [fx, fy]
    z_near: (B)
    z_far: (B)
    c: (B,2)

    Returns: (B, H, W, 8) camera rays with [origin(3), direction(3), near(1), far(1)]
    -------

    """
    B = extrinsics.shape[0]
    device = extrinsics.device

    focal = intrinsics[:, [0, 1], [0, 1]]
    c = intrinsics[:, [0, 1], [-1, -1]]

    pcoords_screen = torch.stack(torch.meshgrid(torch.arange(.5, H, 1, device=device),
                                                torch.arange(0.5, W, 1, device=device))[::-1],
                                 dim=-1)  # (H, W, 2) [x,y] (OPENCV CONVENTION)
    pcoords_screen = pcoords_screen.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
    pcoords_cam = (pcoords_screen - c.view(B, 1, 1, 2)) / focal.view(B, 1, 1, 2)  # (B, H, W, 2)
    pcoords_cam = torch.cat((pcoords_cam, torch.ones_like(pcoords_cam[..., :1])), dim=-1)  # (B, H, W, 3)
    raydirs_cam = pcoords_cam / pcoords_cam.pow(2).sum(dim=-1, keepdim=True).sqrt()

    # rotating view directions into world space
    rots_cam2world = extrinsics[:, :3, :3].permute(0, 2, 1)
    raydirs_world = (rots_cam2world @ raydirs_cam.view(B, -1, 3).permute(0, 2, 1)).permute(0, 2, 1).view(B, H, W, 3)

    # getting ray origins
    cam_centers = (-1 * rots_cam2world @ extrinsics[:, :3, -1:])  # (B, 3, 1)
    ray_origins = cam_centers.view(B, 1, 1, 3).expand(-1, H, W, -1)  # (B, H, W, 3)

    ray_near = z_near.view(B, 1, 1, 1).expand(-1, H, W, -1)
    ray_far = z_far.view(B, 1, 1, 1).expand(-1, H, W, -1)

    rays = torch.cat((ray_origins, raydirs_world, ray_near, ray_far), dim=-1)
    return rays


def trans_t(t):
    return torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1], ], dtype=torch.float32,
    )


def rot_phi(phi):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def rot_theta(th):
    return torch.tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def pose_spherical(theta, phi, radius):
    """
    Spherical rendering poses, from NeRF
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
            torch.tensor(
                [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            @ c2w
    )
    return c2w


def get_ray_intersections(ray1, ray2):
    """
    calculates points on ray1 and ray2 where both rays are closest to another
    :param ray1: torch Tensor [orgx, orgy, orgz, dirx, diry, dirz]
    :param ray2: torch Tensor [orgx, orgy, orgz, dirx, diry, dirz]
    :return:
    """

    B = (ray2[:3] - ray1[:3]).unsqueeze(1)
    A = torch.stack((ray1[3:], -ray2[3:]), dim=-1)

    t1t2 = torch.linalg.lstsq(A, B).solution
    t1t2 = t1t2.flatten()

    x1 = ray1[:3] + ray1[3:] * t1t2[0]
    x2 = ray2[:3] + ray2[3:] * t1t2[1]

    return x1, x2


def to_homogeneous_trafo(trafo: torch.Tensor):
    """

    :param trafo: N, 3, 4
    :return: trafo N, 4, 4 (appended [0,0,0,1])
    """
    return torch.cat((trafo, torch.tensor([[[0, 0, 0, 1.]]]).expand(len(trafo), -1, -1)), dim=1)

class Slerp:
    """
    extends scipy.spatial.transform.Slerp by translation interpolation
    """

    def __init__(self, times, rotations, locations):
        """

        Parameters
        ----------
        times
        rotations
        translations
        """

        self._rotslerp = RotSlerp(times, rotations)
        self._locslerp = TransSlerp(times, locations)

    def __call__(self, times):
        rotations = self._rotslerp(times)
        locations = self._locslerp(times)

        return rotations, locations


class TransSlerp:
    """
    same as scipy.spatial.transform.Slerp but for translations
    """

    def __init__(self, times, locations):
        """

        Parameters
        ----------
        times: fixed times (Nf,)
        translations: fixed locations (Nf,)
        """
        # sort according to times
        idcs = np.argsort(times)
        self._times = times[idcs]
        self._locations = locations[idcs]

    def __call__(self, t_q):
        """
        returns interpolated locations at given times

        Parameters
        ----------
        t_q: query times np.ndarray (Nq,)

        Returns: translations: np.ndarray (Nq, 3)
        -------

        """

        # finding interpolation neighbors
        q_times = np.clip(t_q, a_min=np.min(self._times), a_max=np.max(self._times))
        earlier_fixtimes_mask = q_times[:, None] >= self._times[None]
        later_fixtimes_mask = q_times[:, None] <= self._times[None]
        idx_helper = np.meshgrid(np.arange(len(self._times)), np.arange(len(q_times)))[0]
        earlier_fixpoint_idcs = np.copy(idx_helper)
        later_fixpoint_idcs = np.copy(idx_helper)
        earlier_fixpoint_idcs[~earlier_fixtimes_mask] = 0
        later_fixpoint_idcs[~later_fixtimes_mask] = len(self._times)
        earlier_fixpoint_idcs = np.max(earlier_fixpoint_idcs, axis=1)  # (Nq,)
        later_fixpoint_idcs = np.min(later_fixpoint_idcs, axis=1)  # (Nq, )

        # calculating interpolation weights
        t_earlier = self._times[earlier_fixpoint_idcs]
        t_later = self._times[later_fixpoint_idcs]
        dt = np.clip(t_later - t_earlier, a_min=1e-4, a_max=None)

        w_earlier = np.clip((t_later-t_q) / dt, a_min=0, a_max=1.)
        w_later = 1 - w_earlier

        loc_q = self._locations[earlier_fixpoint_idcs] * w_earlier[:, None] \
                + self._locations[later_fixpoint_idcs] * w_later[:, None]

        return loc_q


if __name__ == "__main__":
    times = np.array([0., 1.])
    locations = np.array([[1., 0, 0], [0., 1., 0]])
    slerp = TransSlerp(times, locations)
    print(slerp(np.linspace(-.1, 1.1, 13)))

    # ray1 = torch.tensor([1, 0, 0, -1, 0, 0.])
    # ray2 = torch.tensor([0, -1, 0, 0, 1. / np.sqrt(2), 1 / np.sqrt(2)])
    # print(get_ray_intersections(ray1, ray2))
