from typing import NamedTuple
import numpy as np
import torch
from torch import nn

from utils.general_utils import PILtoTorch
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_name: str
    width: int
    height: int


class CameraTorch(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        # gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    ):
        super(CameraTorch, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        # self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones(
                (1, self.image_height, self.image_width), device=self.data_device
            )

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


# def loadCam(id, cam_info, resolution_scale):
#     return CameraTorch(
#         colmap_id=cam_info.uid,
#         R=cam_info.R,
#         T=cam_info.T,
#         FoVx=cam_info.FovX,
#         FoVy=cam_info.FovY,
#         gt_alpha_mask=loaded_mask,
#         image_name=cam_info.image_name,
#         uid=id,
#     )


def cameraList_from_camInfos(cam_infos):
    camera_list = []

    for id, cam_info in enumerate(cam_infos):
        camera_list.append(
            CameraTorch(
                colmap_id=cam_info.uid,
                R=cam_info.R,
                T=cam_info.T,
                FoVx=cam_info.FovX,
                FoVy=cam_info.FovY,
                # gt_alpha_mask=loaded_mask,
                image_name=cam_info.image_name,
                uid=id,
                # data_device=args.data_device,
            )
        )

    return camera_list
