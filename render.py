import os
import math
import argparse
from typing import NamedTuple
from os import makedirs
from argparse import ArgumentParser
from typing import List
import torch
import torchvision
import json
import numpy as np
from tqdm import tqdm
from scene import Scene
from utils.general_utils import safe_state
from gaussian_renderer import GaussianModel
from utils.graphics_utils import focal2fov, fov2focal
from utils.system_utils import parse_dimensions
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


# this file is used to render imgs from given sequence of extrinsic and intrinsic


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_name: str


def construct_camera_info_from_transforms(
    transforms_json_file: str,
    image_size: list = [800, 800],
) -> List[CameraInfo]:
    """a simple version of the function readCamerasFromTransforms

    Args:
        path (str):
        transformfile (str):

    Returns:
        CameraInfo:
    """
    cam_infos = []

    with open(transforms_json_file) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # cam_name = os.path.join(path, frame["file_path"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            fovy = focal2fov(fov2focal(fovx, image_size[0]), image_size[1])
            FovY = fovy
            FovX = fovx

            name = frame["file_path"].rsplit("/", 1)[-1]

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image_name=name,
                )
            )

    return cam_infos


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    """
    This code is modified from 3DGS, but way more simpler.

    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FovX * 0.5)
    tanfovy = math.tan(viewpoint_camera.FovY * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def render_from_views(
    # pipe: PipelineParams,
    gs_path: str,
    view_path: str,
    output_path: str,
) -> None:
    if not os.path.exists(view_path):
        raise ValueError(f"path {view_path} does not exist")

    class PIPE:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False

    pipe = PIPE()

    train_cam_infos = construct_camera_info_from_transforms(view_path)

    print("Reading Test Transforms")
    with torch.no_grad():
        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        print(f"gs model loaded from {gs_path}")

        # scene = Scene(dataset, gaussians, shuffle=False)

        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        for id, view in enumerate(train_cam_infos):
            render_img = render(view, gaussians, pipe, background)["render"]

            torchvision.utils.save_image(
                render_img, os.path.join(output_path, "{0:05d}".format(id) + ".png")
            )


def evaluate_args(args):
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    if not os.path.exists(args.model_path):
        raise ValueError(f"model path {args.model_path} not existed")
    if not os.path.exists(args.camera_pose_json):
        raise ValueError(f"model path {args.camera_pose_json} not existed")

    return parse_dimensions(args.image_shape)


if __name__ == "__main__":
    ROOT = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_path", type=str, default=os.path.join(ROOT, "output")
    )
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument(
        "-c",
        "--camera_pose_json",
        type=str,
        default=os.path.join(ROOT, "transforms_train.json"),
    )
    parser.add_argument(
        "-i",
        "--image_shape",
        type=str,
        default="800x800",
        help="e.g. 500x500. WARNING: the size should be align to your json file!",
    )
    args = parser.parse_args()
    img_shape = evaluate_args(args)

    print(args, img_shape)
    exit()

    render_from_views(args, img_shape)
