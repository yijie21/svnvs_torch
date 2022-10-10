import torch
import torch.nn.functional as F
from kornia import create_meshgrid

# global parameter to avoid re-computation
meshgrid_cache = {}


def meshgrid_abs_torch(batch, height, width, device, permute):
    """
    Create a 2D meshgrid with absolute homogeneous coordinates
    Args:
        batch: batch size
        height: height of the grid
        width: width of the grid
    """
    global meshgrid_cache
    # avoid cache size being too large
    if len(meshgrid_cache) > 20:
        meshgrid_cache = {}
    key = (batch, height, width, device, permute)
    try:
        res = meshgrid_cache[key]
    except KeyError:
        grid = create_meshgrid(height, width, device=device, normalized_coordinates=False)[0]
        xs, ys = grid.unbind(-1)
        ones = torch.ones_like(xs)
        coords = torch.stack([xs, ys, ones], axis=0)                        # (3, H, W)
        res = coords[None, ...].repeat(batch, 1, 1, 1).to(device=device)    # (B, 3, H, W)
        if permute:
            res = res.permute(0, 2, 3, 1)                                   # (B, H, W, 3)
        meshgrid_cache[key] = res
    return res

def pixel2cam_torch(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """
    Convert pixel coordinates to camera coordinates (i.e. 3D points).
    Args:
        depth: depth maps -- [B, H, W]
        pixel_coords: pixel coordinates -- [B, 3, H, W]
        intrinsics: camera intrinsics -- [B, 3, 3]
    Returns:
        cam_coords: points in camera coordinates -- [B, 3 (4 if homogeneous), H, W]
    """
    B, H, W = depth.shape
    depth = depth.reshape(B, 1, -1)
    pixel_coords = pixel_coords.reshape(B, 3, -1)
    cam_coords = torch.matmul(torch.inverse(intrinsics), pixel_coords) * depth

    if is_homogeneous:
        ones = torch.ones((B, 1, H*W), device=cam_coords.device)
        cam_coords = torch.cat([cam_coords, ones], dim=1)
    cam_coords = cam_coords.reshape(B, -1, H, W)
    return cam_coords


def cam2pixel_torch(cam_coords, proj):
    """
    Convert camera coordinates to pixel coordinates.
    Args:
        cam_coords: points in camera coordinates -- [B, 4, H, W]
        proj: camera intrinsics -- [B, 4, 4]
    Returns:
        pixel_coords: points in pixel coordinates -- [B, H, W, 2]
    """
    B, _, H, W = cam_coords.shape
    cam_coords = torch.reshape(cam_coords, [B, 4, -1])
    unnormalized_pixel_coords = torch.matmul(proj, cam_coords)
    xy_u = unnormalized_pixel_coords[:, :2, :]
    z_u = unnormalized_pixel_coords[:, 2:3, :]
    pixel_coords = xy_u / (z_u + 1e-10)     # safe division
    pixel_coords = torch.reshape(pixel_coords, [B, 2, H, W])
    return pixel_coords.permute(0, 2, 3, 1)


def resampler_wrapper_torch(images, coords):
    """
    equivalent to tfa.image.resampler
    Args:
        images: [B, H, W, C]
        coords: [B, H, W, 2] source pixel coords 
    """
    return F.grid_sample(
        images,
        torch.tensor([-1, -1], device=images.device) + 2. * coords,
        align_corners=True
    )


def project_inverse_warp_torch(imgs, depth_map, poses, src_intrinsics, tgt_intrinsics, tgt_height, tgt_width):
    """
    Inverse warp a source image to the target image plane based on projection
    Args:
        imgs: source images (n_src_imgs, H, W, 3)
        depth_map: depth map of the target image (n_src_imgs, H, W)
        poses: target to source camera transformation (n_src_imgs, 4, 4)
        src_intrinsics: source camera intrinsics (n_src_imgs, 3, 3)
        tgt_intrinsics: target camera intrinsics (n_src_imgs, 3, 3)
        tgt_height: target image height
        tgt_width: target image width
    Returns:
        imgs_warped: warped images (n_src_imgs, H, W, 3)
    """
    n_src_imgs, _, H, W = imgs.shape
    pixel_coords_N3HW = meshgrid_abs_torch(n_src_imgs, tgt_height, tgt_width, imgs.device, False) # (B, 3, H, W)
    cam_coords_N4HW = pixel2cam_torch(depth_map, pixel_coords_N3HW, tgt_intrinsics)

    # Construct a 4 x 4 intrinsic matrix
    src_intrinsics4 = torch.zeros(n_src_imgs, 4, 4, device=imgs.device)
    src_intrinsics4[:, :3, :3] = src_intrinsics
    src_intrinsics4[:, 3, 3] = 1

    proj_tgt_cam_to_src_pixel = torch.matmul(src_intrinsics4, poses)
    src_pixel_coords_NHW2 = cam2pixel_torch(cam_coords_N4HW, proj_tgt_cam_to_src_pixel)
    
    src_pixel_coords_NHW2 = src_pixel_coords_NHW2 / torch.tensor([W - 1, H - 1], device=imgs.device)

    # mask
    mask_NHW = (src_pixel_coords_NHW2[..., 0] >= 0) & (src_pixel_coords_NHW2[..., 0] <= H) & \
              (src_pixel_coords_NHW2[..., 1] >= 0) & (src_pixel_coords_NHW2[..., 1] <= W)

    output_imgs = resampler_wrapper_torch(imgs, src_pixel_coords_NHW2)
    return output_imgs, mask_NHW


def plane_sweep_torch(imgs, mpi_depths, poses, src_intrinsics, tgt_intrinsics, tgt_height, tgt_width):
    """
    Construct a plane sweep volume
    
    Args:
        imgs: source images (n_src_imgs, h, w, c)
        depth_planes: a list of depth_values for each plane (n_planes, )
        poses: target to source camera transformation (n_src_imgs, 4, 4)
        src_intrinsics: source camera intrinsics (n_src_imgs, 3, 3)
        tgt_intrinsics: target camera intrinsics (n_src_imgs, 3, 3)
        tgt_height: target image height
        tgt_width: target image width
    Returns:
        volume: a tensor of size (n_planes, n_src_imgs, height, width, c)
    """
    n_src_imgs = imgs.shape[0]
    plane_sweep_volume = []
    masks_DNHW = []
    depths = mpi_depths

    for depth in depths:
        curr_depth = torch.zeros([n_src_imgs, tgt_height, tgt_width], dtype=torch.float32, device=imgs.device) + depth
        warped_imgs, mask_NHW = project_inverse_warp_torch(imgs, curr_depth, poses, src_intrinsics, tgt_intrinsics, tgt_height, tgt_width)
        plane_sweep_volume.append(warped_imgs)
        masks_DNHW.append(mask_NHW)
    plane_sweep_volume = torch.stack(plane_sweep_volume, dim=0)
    masks_DNHW = torch.stack(masks_DNHW, dim=0)
    return plane_sweep_volume, masks_DNHW


def stable_softmax(x, dim=-1):
    x = x - torch.max(x, dim=dim, keepdim=True)[0]
    x = x - torch.logsumexp(x, dim=dim, keepdim=True)
    x = torch.exp(x)
    return x