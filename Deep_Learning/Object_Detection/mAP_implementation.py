import torch
from collections import Counter
from IoU_implementation import intersection_over_union

def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners",num_classes=20):
    
    # pred_boxes (list): [[train_idx,class_pred,prob_score,x1,y1,x2,y2],[],[]] (N,7)
    
    average_precision = [] # AP for each class
    episilon = 1e-6

    for c in range(num_classes): # Loop to iterate over each class (1,2,3,...)
        detections = []
        ground_truth = []

        for detection in pred_boxes: # Get all bboxes from predictions where label match with current outer loop label
            if detection[1] == c:
                detections.append(detection)
            
        for true_box in true_boxes:  # Get all bboxes from targets where label match with current outer loop label
            if true_boxes[1] == c:
                ground_truth.append(true_box)


        # img 0 has 3 boxes, img 1 has 5 bboxes so on...
        # amount_bboxes = {0:3,1:5,...}
        amount_bboxes = Counter([gt[0] for gt in ground_truth])


        # amount_boxes = {0:torch.tensor([0,0,0]),1:torch.tensor([0,0,0,0,0]),...}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)


        detections.sort(key=lambda x : x[2],reverse=True) # sort predicted bboxes according to conf

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truth) # total correct bboxes in actuality

        for detection_idx, detection in enumerate(detections): # iterate over each img of predictions

            ground_truth_img = [ # get all true bboxs of one image aswell

                bbox for bbox in ground_truth if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img) # total corect bbox for 1 image
            best_iou = 0

            # get best matching predicted bbbox for each target bbox
            for idx,gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detections[3:]),
                                              torch.tensor(gt[3:]),
                                              box_format=box_format)
                
                if iou >  best_iou:
                    best_iou = iou
                    best_gt_idx = idx # save idx of ground truth

            if best_iou > iou_threshold : # If there is a bbox above iou threshold

                # change for image detection[0] at index best_gt_idx to 1 of it's 0
                # also make TP at index detction_idx to 1
                if amount_bboxes[detection[0]][best_gt_idx] == 0: 
                    TP[detection_idx]=1
                    amount_bboxes[detection[0]][best_gt_idx] = 1

                # if amount_bboxes[detection[0]][best_gt_idx] is already 1,
                # it means that there are multiple bboxes for 1 targert image
                # with iou > threshold
                else:
                    FP[detection_idx] = 1

            # if best_iou < threshold, it means that there is no pred bbox
            # that intersection target bbox with > desired threshold
            # so count it as false positive
            else:
                FP[detection_idx] = 1

        # [1,1,0,1,0] -> [1,2,2,3,3]
        # needed for making graph
        # dim=0 means cumulative sum is computed along the rows 
        TP_cumsum = torch.cumsum(TP,dim=0) 
        FP_cumsum = torch.cumsum(FP,dim=0)

        recalls = TP_cumsum / (total_true_bboxes + episilon) # formula for recall
        precision = torch.divide(TP_cumsum,TP_cumsum+FP_cumsum+episilon) # formula for precision

        precision = torch.cat((torch.tensor[1]),precision)
        recalls = torch.cat((torch.tensor([0])),recalls)

        # torch.trapz performs numerical integration using the trapezoidal rule
        # used to approximate the integral of a function defined by discrete data points
        # first paramter is y, second is x
        # The append all mAP found for each class in average_precision(AP) list
        average_precision.append(torch.trapz(precision,recalls))


    # use this to find mAP for all classes AT A CERTAIN THRESHOLD
    return sum(average_precision) / len(average_precision) 