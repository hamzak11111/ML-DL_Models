import torch

def intersection_over_union(boxes_preds,boxes_labels,box_format="midpoint"):

    # boxes_preds / boxes_labels is of (N,4)

    # boxes_preds[...,0:1]  picks first dimensions which keeping previous dimension same
    # In this context, it is same as boxes_preds[:, 0:1]

    # NOTE: boxes_preds[:, 0:1] is NOT the same as boxes_preds[:, 0]
    # boxes_preds[:, 0:1] returns size of (N,1) [eg: [[1], [2], [3], [4]]]
    # boxes_preds[:, 0] returns size of (N,) or (N) [eg: [1, 2, 3, 4]]


    # Description: This format represents a bounding box using the coordinates of its center point along with its width and height.
    # Structure: [center_x, center_y, width, height]
    # Example: [0.5, 0.5, 0.4, 0.2] (a box centered at (0.5, 0.5) with a width of 0.4 and height of 0.2)
    if box_format == "midpoint":
        # Convert midpoint to corners
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2


    # Description: This format represents a bounding box using the coordinates of its top-left and bottom-right corners.
    # Structure: [x1, y1, x2, y2]
    # Example: [0.1, 0.1, 0.5, 0.5] (a box with the top-left corner at (0.1, 0.1) and bottom-right corner at (0.5, 0.5))

    if box_format == "corners":
        # Use the provided corners directly
        box1_x1 = boxes_preds[...,0:1] 
        box1_y1 = boxes_preds[...,1:2]
        box1_x2 = boxes_preds[...,2:3]
        box1_y2 = boxes_preds[...,3:4]

        box2_x1 = boxes_labels[...,0:1]
        box2_y1 = boxes_labels[...,1:2]
        box2_x2 = boxes_labels[...,2:3]
        box2_y2 = boxes_labels[...,3:4]

    x1 = torch.max(box1_x1,box2_x1)
    y1 = torch.max(box1_y1,box2_y1)
    x2 = torch.max(box1_x2,box2_x2)
    y2 = torch.max(box1_y2,box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

    return intersection/(box1_area + box2_area)-intersection + 1e-6 # 1e-6 used to avoid division by 0
