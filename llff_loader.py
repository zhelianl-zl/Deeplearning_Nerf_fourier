# llff_loader.py
import os
import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import PerspectiveCameras

def _read_poses_bounds(poses_bounds_path):
    poses_bounds = np.load(poses_bounds_path)  # shape (N, 17) for NeRF LLFF, or (N, 3, 5)+bounds in other dumps
    # 兼容两种常见格式：
    if poses_bounds.ndim == 2 and poses_bounds.shape[1] == 17:
        # 每行: 3x5(=15) + 2 = 17
        N = poses_bounds.shape[0]
        poses = poses_bounds[:, :15].reshape(N, 3, 5)      # (N,3,5)
        bounds = poses_bounds[:, 15:]                      # (N,2)
        H = poses[0, 0, 4]
        W = poses[0, 1, 4]
        F = poses[0, 2, 4]  # focal length (pixels)
        return poses, bounds, int(H), int(W), float(F)
    elif poses_bounds.ndim == 3 and poses_bounds.shape[1:] == (3, 5):
        # 一些工具保存的 “poses (N,3,5)”；没有 bounds 则伪造
        poses = poses_bounds
        N = poses.shape[0]
        H = poses[0, 0, 4]
        W = poses[0, 1, 4]
        F = poses[0, 2, 4]
        bounds = np.stack([np.ones(N)*0.1, np.ones(N)*5.0], axis=-1)  # dummy
        return poses, bounds, int(H), int(W), float(F)
    else:
        raise ValueError(f"Unrecognized poses_bounds format: shape={poses_bounds.shape}")

def _load_image(path, target_size=None):
    im = Image.open(path).convert("RGB")
    if target_size is not None:
        Ht, Wt = int(target_size[0]), int(target_size[1])
        im = im.resize((Wt, Ht), Image.LANCZOS)
    im = np.asarray(im).astype(np.float32) / 255.0
    # [H,W,3] -> torch [1,3,H,W]（与你现有 pipeline 对齐）
    im = torch.from_numpy(im).permute(2,0,1).unsqueeze(0)
    return im

def _c2w_to_RT(c2w):
    """
    LLFF pose 是 camera-to-world。Pytorch3D 需要 world-to-camera：
    R = c2w[:3,:3]^T, T = -R @ c2w[:3,3]
    """
    R_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3]
    R = R_c2w.T
    T = -R @ t_c2w
    return R, T

def load_llff_from_poses_bounds(images_dir, poses_bounds_path, downscale=1, device="cuda", target_size=None):
    poses, bounds, H0, W0, F0 = _read_poses_bounds(poses_bounds_path)

    # 按 downscale 缩小
    # downscale
    H = int(H0 / downscale)
    W = int(W0 / downscale)
    fx = F0 / downscale
    fy = F0 / downscale

    # principal point 设在图像中心（像素）
    # The principal point is set at the center of the image (pixel)
    cx = W * 0.5
    cy = H * 0.5

    # 读取 images_dir 下的图片，按自然排序
    # Read the images under images_dir and sort them by natural order
    names = [n for n in os.listdir(images_dir) if n.lower().endswith((".jpg",".jpeg",".png"))]
    def _num_key(s):
        stem = os.path.splitext(s)[0]
        tail = ''.join(ch for ch in stem[::-1] if ch.isdigit())[::-1]
        return int(tail) if tail else float('inf')
    names.sort(key=lambda x: (_num_key(x), x))

    n_img = len(names)
    assert n_img == poses.shape[0], f"images count ({n_img}) != poses count ({poses.shape[0]})"

    samples = []
    for i, name in enumerate(names):
        path = os.path.join(images_dir, name)
        image = _load_image(path, target_size=(H, W))

        # 从 pose 取 3x4 的 c2w
        # Get 3x4 c2w from pose : poses_bounds.npy
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :4] = poses[i, :, :4]

        # ★ 坐标系修正：LLFF(OpenGL) -> OpenCV/PyTorch3D
        # Coordinate system correction: LLFF (OpenGL) -> OpenCV/PyTorch3D
        # Assemble a 4x4 c2w and perform coordinate correction 
        # (LLFF/OpenGL → OpenCV/PyTorch3D): c2w[:3,1]*=-1, c2w[:3,2]*=-1.
        # _c2w_to_RT(c2w) → R = c2w[:3,:3]^T; T = -R @ c2w[:3,3] (obtaining world→camera)

        c2w[:3, 1] *= -1.0
        c2w[:3, 2] *= -1.0

        R, T = _c2w_to_RT(c2w)


        # Pytorch3D 的 focal_length / principal_point 需要**归一化到 NDC**或用像素？
        # 这里使用像素定义（与项目其他地方保持一致）
        focal = torch.tensor([[fx, fy]], dtype=torch.float32, device=device)
        principal_point = torch.tensor([[cx, cy]], dtype=torch.float32, device=device)

        # Creating a Pytorch3D Camera
        cam = PerspectiveCameras(
            # Pixel unit
            focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
            # Pixel unit, image center
            principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
            image_size=torch.tensor([[H, W]], dtype=torch.float32, device=device),
            R=torch.from_numpy(R).unsqueeze(0).to(device=device, dtype=torch.float32),
            T=torch.from_numpy(T).unsqueeze(0).to(device=device, dtype=torch.float32),
            in_ndc=False,   # Very important： use pixel coordinate system
            device=device,
        )

        near_i, far_i = float(bounds[i, 0]), float(bounds[i, 1])

        # Record the near and far values ​​for each image (from poses_bounds.npy)
        # Returns a list of samples: Each element in the dictionary:
        samples.append({
            "image": image.to(device),
            "camera": cam,
            "camera_idx": i,
            "near": torch.tensor([near_i], dtype=torch.float32, device=device),
            "far":  torch.tensor([far_i],  dtype=torch.float32, device=device),
        })


    return samples
