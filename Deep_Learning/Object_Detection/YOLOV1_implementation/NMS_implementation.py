import torch
from IoU_implementation import intersection_over_union

def non_max_suppression(bboxes,iou_threshold,prob_threshold,box_format="corners"):
    # bboxes = [[1,0.9,x1,y1,x2,y2],[],[],...]
    # shape of predictions is (N,6)
    # first index tells label, second confidence, then the 4 points of bbox

    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes,key=lambda x:x[1], reverse=True) # descending order of confidence
    bboxes_after_nms = []
    
    while bboxes:
        chosen_box = bboxes.pop(0) # removes first element with highest confidence

        # this will remove those bboxes that are of the same class AND IOU > threshold (compared to chosen class)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0] # if they are not in same class, keep them

            # or is their iou < threshold, keep them
            or intersection_over_union(torch.tensor(chosen_box[2:]),torch.tensor(box[2:]),box_format=box_format) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)