import torch
import torchvision
from tqdm.auto import tqdm

import common_utils
from common_utils.image import get_ssim_all, get_ssim_pairs_kornia
from evaluations import l2_dist, ncc_dist, normalize_batch, transform_vmin_vmax_batch

import torch.nn.functional as F

def _downscale_by(t: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Downscale NCHW by `factor` (2 or 4).
    On MPS: use avg_pool2d (works on MPS). Else: bicubic interpolate (original behavior).
    """
    assert factor in (2, 4), "factor must be 2 or 4"
    if t.device.type == "mps":
        # MPS doesn't support bicubic; also ensure supported dtype
        t = t.to(torch.float32)
        if t.ndim != 4:
            # Fallback: do resize on CPU then return to MPS
            t_cpu = t.to("cpu")
            t_small = F.interpolate(t_cpu, scale_factor=1.0/factor, mode="bicubic", align_corners=False)
            return t_small.to("mps")
        return F.avg_pool2d(t, kernel_size=factor, stride=factor)
    else:
        return F.interpolate(t, scale_factor=1.0/factor, mode="bicubic", align_corners=False)



@torch.no_grad()
def get_dists(x, y, search, use_bb):
    """D: x -> y"""
    xxx = x.clone()
    yyy = y.clone()

    # Search Real --> Extracted
    if search == 'l2':
        D = l2_dist(xxx, yyy, div_dim=True)
    if search == 'ncc':
        D = ncc_dist(xxx, yyy, div_dim=True)
    elif search == 'ncc2':
        x2search = torch.nn.functional.interpolate(xxx, scale_factor=1 / 2, mode='bicubic', align_corners=False)
        y2search = torch.nn.functional.interpolate(yyy, scale_factor=1 / 2, mode='bicubic', align_corners=False)
        D = ncc_dist(x2search, y2search, div_dim=True)
    elif search == 'ncc4':
        x2search = torch.nn.functional.interpolate(xxx, scale_factor=1/4, mode='bicubic', align_corners=False)
        y2search = torch.nn.functional.interpolate(yyy, scale_factor=1/4, mode='bicubic', align_corners=False)
        D = ncc_dist(x2search, y2search, div_dim=True)

    # Consider each reconstruction for only one train-samples
    if use_bb:
        bb_mask = D.mul(-100000000).softmax(dim=0).mul(10).round().div(10).round()
        assert bb_mask.sum(dim=0).abs().sum() == D.shape[1]
        D[bb_mask != 1] = torch.inf

    dists, idxs = D.sort(dim=1, descending=False)
    return dists, idxs


@torch.no_grad()
def find_nearest_neighbour(X, x0, search='ncc', vote='mean', use_bb=True, nn_threshold=None, ret_idxs=False):
    xxx = X.clone()
    yyy = x0.clone()

    # Ensure MPS-safe dtype if needed
    if xxx.device.type == "mps":
        xxx = xxx.to(torch.float32)
        yyy = yyy.to(torch.float32)

    # Search Real --> Extracted
    if search == 'l2':
        D = l2_dist(yyy, xxx, div_dim=True)
    if search == 'ncc':
        D = ncc_dist(yyy, xxx, div_dim=True)

    elif search == 'ncc2':
        x2search = _downscale_by(xxx, 2)
        y2search = _downscale_by(yyy, 2)
        # --- CPU fallback for ncc on MPS to avoid einsum bug ---
        if x2search.device.type == "mps":
            D = ncc_dist(y2search.to("cpu").contiguous(),
                         x2search.to("cpu").contiguous(),
                         div_dim=True).to("mps")
        else:
            D = ncc_dist(y2search, x2search, div_dim=True)

    elif search == 'ncc4':
        x2search = _downscale_by(xxx, 4)
        y2search = _downscale_by(yyy, 4)
        # --- CPU fallback for ncc on MPS to avoid einsum bug ---
        if x2search.device.type == "mps":
            D = ncc_dist(y2search.to("cpu").contiguous(),
                         x2search.to("cpu").contiguous(),
                         div_dim=True).to("mps")
        else:
            D = ncc_dist(y2search, x2search, div_dim=True)

    elif search == 'dssim':
        D_ssim = get_ssim_all(yyy, xxx)
        D = (1 - D_ssim) / 2

    # Only consider Best-Bodies
    if use_bb:
        bb_mask = D.mul(-100000000).softmax(dim=0).mul(10).round().div(10).round()
        assert bb_mask.sum(dim=0).abs().sum() == D.shape[1]
        D[bb_mask != 1] = torch.inf

    dists, idxs = D.sort(dim=1, descending=False)

    if vote == 'min' or vote is None:
        xx = xxx[idxs[:, 0]]
    else:
        if nn_threshold is None:
            xs_idxs = idxs[:, :int(0.01 * x0.shape[0])]
        else:
            xs_idxs = []
            for i in range(dists.shape[0]):
                x_idxs = [idxs[i, 0].item()]
                for j in range(1, dists.shape[1]):
                    if (dists[i, j] / dists[i, 0]) < nn_threshold:
                        x_idxs.append(idxs[i, j].item())
                    else:
                        break
                xs_idxs.append(x_idxs)

        xs = []
        for x_idxs in xs_idxs:
            if vote == 'min':
                x_voted = xxx[x_idxs[0]].unsqueeze(0)
            elif vote == 'mean':
                x_voted = xxx[x_idxs].mean(dim=0, keepdim=True)
            elif vote == 'median':
                x_voted = xxx[x_idxs].median(dim=0, keepdim=True).values
            elif vote == 'mode':
                x_voted = xxx[x_idxs].mode(dim=0, keepdim=True).values
            else:
                raise
            xs.append(x_voted)
        xx = torch.cat(xs, dim=0).clone()

    if ret_idxs:
        return xx, idxs[:, 0]
    return xx


@torch.no_grad()
def scale(xx, x0, ds_mean, xx_add_ds_mean=True):
    dev = xx.device

    def _to_cpu64(t: torch.Tensor) -> torch.Tensor:
        # First move device, then cast dtype (avoids MPS fp64 path)
        if t.device.type == "mps":
            t_cpu = t.detach().to("cpu")          # keep original dtype
            return t_cpu.to(torch.float64)        # cast on CPU
        else:
            return t.detach().to(dtype=torch.float64)

    # Do numerically sensitive ops on CPU/float64
    xx_cpu = _to_cpu64(xx)
    x0_cpu = _to_cpu64(x0)
    dm_cpu = _to_cpu64(ds_mean)

    yy_cpu = x0_cpu + dm_cpu
    if xx_add_ds_mean:
        xx_cpu = transform_vmin_vmax_batch(xx_cpu + dm_cpu)
    else:
        xx_cpu = transform_vmin_vmax_batch(xx_cpu)

    # Move results back to the original device
    if dev.type == "mps":
        xx_out = xx_cpu.to(dev, dtype=torch.float32)
        yy_out = yy_cpu.to(dev, dtype=torch.float32)
    else:
        xx_out = xx_cpu.to(dev, dtype=xx.dtype)
        yy_out = yy_cpu.to(dev, dtype=x0.dtype)

    return xx_out, yy_out





@torch.no_grad()
def sort_by_metric(xx, yy, sort='ssim'):
    xx = xx.clone()
    yy = yy.clone()

    # Score
    psnr = lambda a, b: 20 * torch.log10(1.0 / (a - b).pow(2).reshape(a.shape[0], -1).mean(dim=1).sqrt())

    # Sort
    if sort == 'ssim':
        dists = get_ssim_pairs_kornia(xx, yy)
        dssim = (1 - dists) / 2
        _, sort_idxs = dists.sort(descending=True)
    elif sort == 'ncc':
        dists = (normalize_batch(xx) - normalize_batch(yy)).reshape(xx.shape[0], -1).norm(dim=1)
        _, sort_idxs = dists.sort()
    elif sort == 'l2':
        dists = (xx - yy).reshape(xx.shape[0], -1).norm(dim=1)
        _, sort_idxs = dists.sort()
    elif sort == 'psnr':
        dists = psnr(xx, yy)
        _, sort_idxs = dists.sort(descending=True)
    else:
        raise

    xx = xx[sort_idxs]
    yy = yy[sort_idxs]
    return xx, yy, dists, sort_idxs


@torch.no_grad()
def plot_table(xx, yy, fig_elms_in_line, fig_lines_per_page, fig_type='side_by_side',
               figpath=None, show=False, dpi=100, color_by_labels=None):
    # PRINT TABLES
    import matplotlib.pyplot as plt
    xx = xx.clone()
    yy = yy.clone()

    RED = torch.tensor([1, 0, 0])[None, :, None, None]
    BLUE = torch.tensor([0, 1, 0])[None, :, None, None]
    def add_colored_margin(x, labels, p=1):
        n, c, h, w = x.shape
        bg = torch.zeros(n, c, h + 2 * p, w + 2 * p)
        bg[labels == 0] += RED
        bg[labels == 1] += BLUE
        bg[:, :, p:-p, p:-p] = x
        return bg

    if color_by_labels is not None:
        yy = add_colored_margin(yy, color_by_labels, p=2)
        xx = add_colored_margin(xx, color_by_labels, p=2)

    if fig_type == 'side_by_side':
        qq = torch.stack(common_utils.common.flatten(list(zip(xx, yy))))
    elif fig_type == 'one_above_another':
        q_zip = common_utils.common.flatten(list(zip(torch.split(xx, fig_elms_in_line), torch.split(yy, fig_elms_in_line))))
        if len(q_zip) > 2:
            q_zip = q_zip[:-2]
            print('CUT the end of the zipped bla because it might have different shape before torch.cat')
        qq = torch.cat(q_zip)
    else:
        raise

    lines_num = qq.shape[0] // fig_elms_in_line
    print(qq.shape, lines_num)
    for page_num, line_num in enumerate(tqdm(range(0, lines_num, fig_lines_per_page))):
        s = line_num * fig_elms_in_line
        e = (line_num + fig_lines_per_page) * fig_elms_in_line
        print(page_num, s, e)
        grid = torchvision.utils.make_grid(qq[s:e], normalize=False, nrow=fig_elms_in_line, pad_value=1)
        if figpath is not None:
            plt.imsave(figpath, grid.permute(1, 2, 0).cpu().numpy(), dpi=dpi)
            print('Saved fig at:', figpath)
        if show:
            plt.figure(figsize=(80 * 2, 10 * 2))
            plt.axis('off')
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.show()
        plt.close('all')
        break
    print('DONE!')
