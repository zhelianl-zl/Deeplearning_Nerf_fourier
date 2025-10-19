import os
import warnings

import hydra
import numpy as np
import torch
import tqdm
import imageio

from omegaconf import DictConfig
from PIL import Image
from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform
)
import matplotlib.pyplot as plt

from implicit import volume_dict
from sampler import sampler_dict
from renderer import renderer_dict
from ray_utils import (
    sample_images_at_xy,
    get_pixels_from_image,
    get_random_pixels_from_image,
    get_rays_from_pixels
)
from data_utils import (
    dataset_from_config,
    create_surround_cameras,
    vis_grid,
    vis_rays,
)
from dataset import (
    get_nerf_datasets,
    trivial_collate,
)

from render_functions import render_points


# Model class containing:
#   1) Implicit volume defining the scene
#   2) Sampling scheme which generates sample points along rays
#   3) Renderer which can render an implicit volume given a sampling scheme

class Model(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        # Get implicit function from config
        self.implicit_fn = volume_dict[cfg.implicit_function.type](
            cfg.implicit_function
        )

        # Point sampling (raymarching) scheme
        self.sampler = sampler_dict[cfg.sampler.type](
            cfg.sampler
        )

        # Initialize volume renderer
        self.renderer = renderer_dict[cfg.renderer.type](
            cfg.renderer
        )
    
    def forward(
        self,
        ray_bundle
    ):
        # Call renderer with
        #  a) Implicit volume
        #  b) Sampling routine

        return self.renderer(
            self.sampler,
            self.implicit_fn,
            ray_bundle
        )


def render_images(
    model,
    cameras,
    image_size,
    save=True,
    file_prefix='images/flower',
    nears=None,            # ← 新增
    fars=None              # ← 新增
):
    all_images = []
    device = list(model.parameters())[0].device

    for cam_idx, camera in enumerate(cameras):
        print(f'Rendering image {cam_idx}')
        torch.cuda.empty_cache()
        camera = camera.to(device)

        xy_grid   = get_pixels_from_image(image_size, camera)
        ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera)

        # 渲染时也写入该视角的 near/far（若提供）
        if nears is not None and fars is not None:
            ray_bundle.nears = nears[cam_idx]
            ray_bundle.fars  = fars[cam_idx]

        # 采样点
        ray_bundle = model.sampler(ray_bundle)

        # 前向渲染
        out = model(ray_bundle)

        # === DEBUG: 渲染输出体检（可留可删）===
        feat = out['feature']
        if not torch.isfinite(feat).all():
            print("[WARN] feature contains NaN/Inf")
        print("render stats: min/mean/max =",
              float(feat.min()), float(feat.mean()), float(feat.max()))
        if 'depth' in out:
            d = out['depth']
            print("depth stats: min/mean/max =",
                  float(d.min()), float(d.mean()), float(d.max()))
        # ==================================

        image = np.array(out['feature'].view(image_size[1], image_size[0], 3).detach().cpu())
        all_images.append(image)

        if save:
            os.makedirs(os.path.dirname(file_prefix), exist_ok=True)
            out_path = f'{file_prefix}_{cam_idx:03d}.png'
            plt.imsave(out_path, image)

    return all_images




def render(
    cfg,
):
    # Create model
    model = Model(cfg)
    model = model.cuda(); model.eval()

    # Render spiral
    cameras = create_surround_cameras(3.0, n_poses=20)
    all_images = render_images(
        model, cameras, cfg.data.image_size, save=True
    )
    imageio.mimsave('images/part_1.gif', [np.uint8(im * 255) for im in all_images], loop = 0)


def train(
    cfg
):
    # Create model
    model = Model(cfg)
    model = model.cuda(); model.train()

    # Create dataset 
    train_dataset = dataset_from_config(cfg.data)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )
    image_size = cfg.data.image_size

    # Create optimizer 
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr
    )

    # Render images before training
    cameras = [item['camera'] for item in train_dataset]
    render_images(
        model, cameras, image_size,
        save=True, file_prefix='images/part_2_before_training'
    )

    # Train
    t_range = tqdm.tqdm(range(cfg.training.num_epochs))

    for epoch in t_range:
        for iteration, batch in enumerate(train_dataloader):
            image, camera, camera_idx = batch[0].values()
            image = image.cuda()
            camera = camera.cuda()

            # Sample rays
            xy_grid = get_random_pixels_from_image(cfg.training.batch_size, image_size, camera) # TODO (2.1): implement in ray_utils.py
            ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera)
            rgb_gt = sample_images_at_xy(image, xy_grid)

            from PIL import Image as PILImage
            if epoch == start_epoch and iteration == 0:
                print("rgb_gt mean/std:", float(rgb_gt.mean()), float(rgb_gt.std()))
                dbg = (images_bhwc[0].detach().cpu().numpy() * 255).astype('uint8')
                PILImage.fromarray(dbg).save('images/_debug_input.png')

            # Run model forward
            out = model(ray_bundle)

            assert torch.isfinite(out["feature"]).all(), "model output has NaN/Inf"

            # TODO (2.2): Calculate loss
            loss = torch.mean((out['feature']-rgb_gt)**2)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch % 10) == 0:
            t_range.set_description(f'Epoch: {epoch:04d}, Loss: {loss:.06f}')
            t_range.refresh()

    # Print center and side lengths
    print("Box center:", tuple(np.array(model.implicit_fn.sdf.center.data.detach().cpu()).tolist()[0]))
    print("Box side lengths:", tuple(np.array(model.implicit_fn.sdf.side_lengths.data.detach().cpu()).tolist()[0]))

    # Render images after training
    render_images(
        model, cameras, image_size,
        save=True, file_prefix='images/part_2_after_training'
    )
    all_images = render_images(
        model, create_surround_cameras(3.0, n_poses=20), image_size, file_prefix='part_2'
    )
    imageio.mimsave('images/part_2.gif', [np.uint8(im * 255) for im in all_images], loop=0)


def create_model(cfg):
    # Create model
    model = Model(cfg)
    model.cuda(); model.train()

    # Load checkpoints
    optimizer_state_dict = None
    start_epoch = 0

    checkpoint_path = os.path.join(
        hydra.utils.get_original_cwd(),
        cfg.training.checkpoint_path
    )

    if len(cfg.training.checkpoint_path) > 0:
        # Make the root of the experiment directory.
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume training if requested.
        if cfg.training.resume and os.path.isfile(checkpoint_path):
            print(f"Resuming from checkpoint {checkpoint_path}.")
            loaded_data = torch.load(checkpoint_path)
            model.load_state_dict(loaded_data["model"])
            start_epoch = loaded_data["epoch"]

            print(f"   => resuming from epoch {start_epoch}.")
            optimizer_state_dict = loaded_data["optimizer"]

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
    )

    # Load the optimizer state dict in case we are resuming.
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        optimizer.last_epoch = start_epoch

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    def lr_lambda(epoch):
        return cfg.training.lr_scheduler_gamma ** (
            epoch / cfg.training.lr_scheduler_step_size
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    return model, optimizer, lr_scheduler, start_epoch, checkpoint_path

def train_nerf(cfg):
    from torch.utils.data import Dataset
    from llff_loader import load_llff_from_poses_bounds

    cameras_all = None   # 先占位，保证后面可用

    # ===== 准备数据（并把 near/far 写回 cfg.sampler 用于采样范围）=====
    if cfg.data.dataset_name == "llff_custom":
        H, W = cfg.data.image_size[1], cfg.data.image_size[0]
        samples = load_llff_from_poses_bounds(
            images_dir=cfg.data.images_dir,
            poses_bounds_path=cfg.data.poses_bounds_path,
            downscale=int(cfg.data.downscale),
            device="cuda",
            target_size=(H, W),
        )

        cameras_all = [s["camera"] for s in samples]
        nears_all   = [s["near"]   for s in samples]
        fars_all    = [s["far"]    for s in samples]


        import numpy as np
        all_nears = np.array([s["near"].item() for s in samples])
        all_fars  = np.array([s["far"].item()  for s in samples])
        eps = 1e-3
        cfg.sampler.min_depth = float(max(all_nears.min(), eps))
        cfg.sampler.max_depth = float(max(all_fars.max(), cfg.sampler.min_depth + 1e-3))
        print("Using global near/far:", cfg.sampler.min_depth, cfg.sampler.max_depth)

        class _ListDS(Dataset):
            def __init__(self, items): self.items = items
            def __len__(self): return len(self.items)
            def __getitem__(self, i): return self.items[i]

        n = len(samples)
        n_train = max(1, int(0.9 * n))
        train_dataset = _ListDS(samples[:n_train])
        val_dataset   = _ListDS(samples[n_train:])
    else:
        train_dataset, val_dataset, _ = get_nerf_datasets(
            dataset_name=cfg.data.dataset_name,
            image_size=[cfg.data.image_size[1], cfg.data.image_size[0]],
        )

    # ===== 再创建模型（此时 cfg.sampler 已被 near/far 更新）=====
    model, optimizer, lr_scheduler, start_epoch, checkpoint_path = create_model(cfg)
    print("Nerf model:-")
    print(model)
    # （可选）训练开始前渲染一遍，确认不是全黑
    if cameras_all is not None:
        model.eval()
        with torch.no_grad():
            imgs = render_images(
                model, cameras_all, cfg.data.image_size,
                save=True, file_prefix='images/flower',
                nears=nears_all, fars=fars_all
            )
        model.train()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=trivial_collate,
    )

    # ===== 训练循环 =====
    for epoch in range(start_epoch, cfg.training.num_epochs):
        t_range = tqdm.tqdm(enumerate(train_dataloader))
        for iteration, batch in t_range:
            image = batch[0]["image"].cuda()   # 可能是 [1,3,H,W] 或 [3,H,W]
            camera = batch[0]["camera"].cuda()
            near   = batch[0]["near"]
            far    = batch[0]["far"]

            # 统一成 [B,H,W,3] 以便采样像素
            if image.dim() == 4 and image.shape[1] == 3:        # [B,3,H,W]
                images_bhwc = image.permute(0, 2, 3, 1)
            elif image.dim() == 3 and image.shape[0] == 3:      # [3,H,W]
                images_bhwc = image.permute(1, 2, 0).unsqueeze(0)
            elif image.dim() == 4 and image.shape[-1] == 3:     # [B,H,W,3]
                images_bhwc = image
            else:
                raise RuntimeError(f"Unexpected image shape: {tuple(image.shape)}")


            # 采样像素 & 构造光线
            xy_grid   = get_random_pixels_from_image(cfg.training.batch_size, cfg.data.image_size, camera)
            ray_bundle = get_rays_from_pixels(xy_grid, cfg.data.image_size, camera)

            # 写入 near/far 到 bundle（字段名按你的 RayBundle 定义，这里用 nears/fars）
            ray_bundle.nears = near
            ray_bundle.fars  = far

            # 前向与损失
            rgb_gt = sample_images_at_xy(images_bhwc, xy_grid)  # [K,3]
            out    = model(ray_bundle)                          # out['feature']:[K,3]
            loss   = torch.mean((out["feature"] - rgb_gt) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_range.set_description(f'Epoch: {epoch:04d}, Loss: {loss:.06f}')
            t_range.refresh()

        # 学习率
        lr_scheduler.step()

        # 每隔 render_interval 个 epoch 渲染一次（用你给的全部相机）
        if (epoch % cfg.training.render_interval == 0) and (epoch > 0) and (cameras_all is not None):
            model.eval()
            with torch.no_grad():
                imgs = render_images(
                    model, cameras_all, cfg.data.image_size,
                    save=True, file_prefix='images/flower',
                    nears=nears_all, fars=fars_all
                )
                imageio.mimsave('images/flower.gif', [np.uint8(im*255) for im in imgs], duration=0.08)
            model.train()

        # 按需保存 checkpoint（可留可去）
        if (epoch % cfg.training.checkpoint_interval == 0) and (len(cfg.training.checkpoint_path) > 0) and (epoch > 0):
            print(f"Storing checkpoint {checkpoint_path}.")
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
                checkpoint_path
            )


#@hydra.main(config_path='./configs', config_name='sphere')
@hydra.main(config_path='./configs', config_name='nerf_flower')
def main(cfg: DictConfig):

    from omegaconf import OmegaConf
    print("### EFFECTIVE CFG ###")
    print(OmegaConf.to_yaml(cfg))

    os.chdir(hydra.utils.get_original_cwd())

    if cfg.type == 'render':
        render(cfg)
    elif cfg.type == 'train':
        train(cfg)
    elif cfg.type == 'train_nerf':
        train_nerf(cfg)


if __name__ == "__main__":
    main()
