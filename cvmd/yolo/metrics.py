# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model validation metrics."""

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch



OKS_SIGMA = (
    np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
    / 10.0
)


def bbox_ioa(box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-7) -> np.ndarray:
    """
    Calculate the intersection over box2 area given box1 and box2.

    Args:
        box1 (np.ndarray): A numpy array of shape (N, 4) representing N bounding boxes in x1y1x2y2 format.
        box2 (np.ndarray): A numpy array of shape (M, 4) representing M bounding boxes in x1y1x2y2 format.
        iou (bool, optional): Calculate the standard IoU if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (np.ndarray): A numpy array of shape (N, M) representing the intersection over box2 area.
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate intersection-over-union (IoU) of boxes.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes in (x1, y1, x2, y2) format.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes in (x1, y1, x2, y2) format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.

    References:
        https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4.
    For instance, you may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4).
    Internally, the code will split the last dimension into (x, y, w, h) if `xywh=True`,
    or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format.
        GIoU (bool, optional): If True, calculate Generalized IoU.
        DIoU (bool, optional): If True, calculate Distance IoU.
        CIoU (bool, optional): If True, calculate Complete IoU.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def mask_iou(mask1: torch.Tensor, mask2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate masks IoU.

    Args:
        mask1 (torch.Tensor): A tensor of shape (N, n) where N is the number of ground truth objects and n is the
                        product of image width and height.
        mask2 (torch.Tensor): A tensor of shape (M, n) where M is the number of predicted objects and n is the
                        product of image width and height.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing masks IoU.
    """
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def kpt_iou(
    kpt1: torch.Tensor, kpt2: torch.Tensor, area: torch.Tensor, sigma: List[float], eps: float = 1e-7
) -> torch.Tensor:
    """
    Calculate Object Keypoint Similarity (OKS).

    Args:
        kpt1 (torch.Tensor): A tensor of shape (N, 17, 3) representing ground truth keypoints.
        kpt2 (torch.Tensor): A tensor of shape (M, 17, 3) representing predicted keypoints.
        area (torch.Tensor): A tensor of shape (N,) representing areas from ground truth.
        sigma (list): A list containing 17 values representing keypoint scales.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing keypoint similarities.
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)  # (N, M, 17)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)  # (17, )
    kpt_mask = kpt1[..., 2] != 0  # (N, 17)
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # from cocoeval
    # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # from formula
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)


def _get_covariance_matrix(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate covariance matrix from oriented bounding boxes.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(obb1: torch.Tensor, obb2: torch.Tensor, CIoU: bool = False, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 5), format xywhr.
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 5), format xywhr.
        CIoU (bool, optional): If True, calculate CIoU.
        eps (float, optional): Small value to avoid division by zero.

    Returns:
        (torch.Tensor): OBB similarities, shape (N,).

    Notes:
        OBB format: [center_x, center_y, width, height, rotation_angle].

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou


def batch_probiou(
    obb1: Union[torch.Tensor, np.ndarray], obb2: Union[torch.Tensor, np.ndarray], eps: float = 1e-7
) -> torch.Tensor:
    """
    Calculate the probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def smooth_bce(eps: float = 0.1) -> Tuple[float, float]:
    """
    Compute smoothed positive and negative Binary Cross-Entropy targets.

    Args:
        eps (float, optional): The epsilon value for label smoothing.

    Returns:
        pos (float): Positive label smoothing BCE target.
        neg (float): Negative label smoothing BCE target.

    References:
        https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """
    return 1.0 - 0.5 * eps, 0.5 * eps





def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed






def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    on_plot=None,
    save_dir: Path = Path(),
    names: Dict[int, str] = {},
    eps: float = 1e-16,
    prefix: str = "",
) -> Tuple:
    """
    Compute the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not.
        on_plot (callable, optional): A callback to pass plots path and data when they are rendered.
        save_dir (Path, optional): Directory to save the PR curves.
        names (Dict[int, str], optional): Dictionary of class names to plot PR curves.
        eps (float, optional): A small value to avoid division by zero.
        prefix (str, optional): A prefix string for saving the plot files.

    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.
        fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class.
        p (np.ndarray): Precision values at threshold given by max F1 metric for each class.
        r (np.ndarray): Recall values at threshold given by max F1 metric for each class.
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class.
        ap (np.ndarray): Average precision for each class at different IoU thresholds.
        unique_classes (np.ndarray): An array of unique classes that have data.
        p_curve (np.ndarray): Precision curves for each class.
        r_curve (np.ndarray): Recall curves for each class.
        f1_curve (np.ndarray): F1-score curves for each class.
        x (np.ndarray): X-axis values for the curves.
        prec_values (np.ndarray): Precision values at mAP@0.5 for each class.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = {i: names[k] for i, k in enumerate(unique_classes) if k in names}  # dict: only classes that have data
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values



