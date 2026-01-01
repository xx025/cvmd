import numpy as np


def detect_with_windows(
    image, windows, model, merge=True, merge_iou=0.2
):
    """
    Detect in multiple windows and optionally merge overlapped duplicates.

    model(crop) -> results (N,6) or (results, extra)
    results format: [x1,y1,x2,y2, conf, cls] in window-local coords

    Args:
        image: (H,W,3) np.ndarray
        windows: list of (x1,y1,x2,y2)
        model: function that takes in an image crop and returns detections
        merge: using WBF to merge boxes from different windows
        merge_iou: IoU threshold for merging boxes,default 0.001 to merge almost all overlapping boxes
        class_aware: whether to merge boxes only of the same class
    """
    H, W = image.shape[:2]

    windows = np.asarray(windows).astype(int)
    windows = windows.clip([0, 0, 0, 0], [W, H, W, H])

    part_imgs = [image[y1:y2, x1:x2] for (x1, y1, x2, y2) in windows]
    all_results = []
    for crop, (wx1, wy1, wx2, wy2) in zip(part_imgs, windows):
        results = model(crop)
        pred = results if isinstance(results, np.ndarray) else results[0]
        if pred.shape[0] > 0:
            pred[:, :4] += np.tile((wx1, wy1), 2)
            all_results.append(pred)

    pred = np.vstack(all_results) if all_results else np.empty((0, 6))

    if not merge:
        return pred
    else:
        from ensemble_boxes import weighted_boxes_fusion
        scale = np.array([W, H, W, H], dtype=np.float32)
        pred[:, :4] /= scale
        
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list=[pred[:, :4].tolist()],
            scores_list=[pred[:, 4].tolist()],
            labels_list=[pred[:, 5].astype(int).tolist()],
            iou_thr=merge_iou
        )
        boxes = np.asarray(boxes, dtype=np.float32) * scale
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        return np.hstack((boxes, scores[:, None], labels[:, None]))
