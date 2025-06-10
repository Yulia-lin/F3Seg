import cv2
import random
import json
import time

import torch
import numpy as np

import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide 

from PIL import Image  
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

import wandb
from IPython import embed
from tqdm import tqdm
from datetime import datetime
from utils.utils import dc, jc, hd95, asd
from functools import reduce

import copy

def hyper_params_tuning(sam):

    mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=4,
    points_per_batch=128,
    pred_iou_thresh=0.5,
    stability_score_thresh=0.5,
    box_nms_thresh= 0.8,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    crop_nms_thresh=0.8,
    min_mask_region_area=200, 
    )

    return mask_generator, 50000

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image) 
    return image.permute(2, 0, 1).contiguous()



def get_crops(image, masks, prompt_mode="crops"):
    imgs_bboxes = []
    indices_to_remove = []

    for i, mask in enumerate(masks):
        box = mask["bbox"]
        x1 = box[0]
        y1 = box[1]
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        
        if x2 > x1 and y2 > y1:  # Check if the bounding box has non-zero dimensions

            if prompt_mode == "crops":
                # crops
                seg_mask = np.array([mask["segmentation"], mask["segmentation"], mask["segmentation"]]).transpose(1,2,0)
                cropped_image = np.multiply(image, seg_mask).astype("int")[int(y1):int(y2), int(x1):int(x2)]
                imgs_bboxes.append(cropped_image)
            
            elif prompt_mode == "crop_expand":
                #crops
                seg_mask = np.array([mask["segmentation"], mask["segmentation"], mask["segmentation"]]).transpose(1,2,0)
                # Expand bounding box coordinates
                x1_expanded = max(0, x1 - 10)
                y1_expanded = max(0, y1 - 10)
                x2_expanded = min(image.shape[1], x2 + 10)
                y2_expanded = min(image.shape[0], y2 + 10)
                
                if x2_expanded > x1_expanded and y2_expanded > y1_expanded:
                    cropped_image = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
                    imgs_bboxes.append(cropped_image)
                
            elif prompt_mode == "bbox":
                # bbox on the image around crop area
                img_bbox = cv2.rectangle(image.copy(), (int(x1), int(y1)), (int(x2), int(y2)), ( 255, 0, 0), 5)
                imgs_bboxes.append(img_bbox)

            elif prompt_mode == "reverse_box_mask":
                # highlight roi and gray out the rest
                res = image.copy()
                box_mask = np.zeros(res.shape, dtype=np.uint8)
                box_mask = cv2.rectangle(box_mask, (x1, y1), (x2, y2),
                                        color=(255, 255, 255), thickness=-1)[:, :, 0]
                overlay = res.copy()
                overlay[box_mask == 0] = np.array((124, 116, 104))
                alpha = 0.5 # Transparency factor.
                res = cv2.addWeighted(overlay, alpha, res, 1 - alpha, 0.0)
                imgs_bboxes.append(res)

            elif prompt_mode == "contour":
                
                #contour around the mask and overlay on the image
                res = image.copy()
                mask = mask["segmentation"]
                contours, hierarchy = cv2.findContours(mask.astype(
                        np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                res = cv2.drawContours(res, contours, contourIdx=-1,
                                        color=(255, 0, 0), thickness=3)#, gave 60.04
                imgs_bboxes.append(res)

        else:
            print("Skipping zero-sized bounding box.")
            indices_to_remove.append(i)

            
        for index in sorted(indices_to_remove, reverse=True):
                del masks[index] 

    return imgs_bboxes

def get_top_scores_with_threshold(scores, threshold_multiplier=1):
    mean = sum(scores) / len(scores)  
    std_dev = (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5 
    max_score = max(scores)
    threshold = max_score - threshold_multiplier * std_dev 


    high_indices = [i for i, score in enumerate(scores) if score >= threshold]
    sorted_high_indices = sorted(high_indices, key=lambda i: scores[i], reverse=True)
    return sorted_high_indices

def get_bottom_scores_with_threshold(scores, threshold_multiplier=1):
    mean = sum(scores) / len(scores)  
    std_dev = (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5 
    min_score = min(scores)
    threshold = min_score + threshold_multiplier * std_dev 

    high_indices = [i for i, score in enumerate(scores) if score <= threshold]
    sorted_low_indices = sorted(high_indices, key=lambda i: scores[i], reverse=False)
    return sorted_low_indices

def retrieve_relevant_crop(crops, class_names, model, preprocess, config):
    crops_uint8 = [image.astype(np.uint8) for image in crops]

    pil_images = []
    for image in crops_uint8:
        if image.shape[0] > 0 and image.shape[1] > 0:
            pil_image = Image.fromarray(image)
            pil_images.append(pil_image)
  
    preprocessed_images = [preprocess(image).to("cuda") for image in pil_images]

    if not preprocessed_images:
        return {class_name: [0] for class_name in class_names}, {class_name: [0] for class_name in class_names}
    stacked_images = torch.stack(preprocessed_images)

    similarity_scores = {class_name: [] for class_name in class_names}

    with torch.no_grad():

        image_features = model.encode_image(stacked_images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        for class_name in class_names:
            class_descriptions = class_names[class_name]
            class_text_features = [model.encode_text(clip.tokenize(description).to("cuda")) for description in class_descriptions]

            mean_text_feature = torch.mean(torch.stack(class_text_features), dim=0)
            mean_text_feature /= mean_text_feature.norm(dim=-1, keepdim=True)
            
            similarity_score = 100. * image_features @ mean_text_feature.T
            similarity_scores[class_name] = similarity_score.squeeze().tolist()

        max_indices = {}
        for key in similarity_scores:
            if isinstance(similarity_scores[key], float):
                max_indices[key] = 0
            else:
                if key == 'background':
                    max_indices[key] = get_bottom_scores_with_threshold(similarity_scores[key], threshold_multiplier=1)
                else:
                    max_indices[key] = get_top_scores_with_threshold(similarity_scores[key], threshold_multiplier=1)

    return max_indices, similarity_scores

def filter_array(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return [a] if a == b else [a]

    if isinstance(a, int) and isinstance(b, (list, set, tuple)):
        return [a] if a in b else [a]

    if isinstance(a, (list, set, tuple)) and isinstance(b, int):
        return [x for x in a if x == b] if b in a else list(a)

    if isinstance(a, (list, set, tuple)) and isinstance(b, (list, set, tuple)):
        intersection = [x for x in a if x in b]
        return intersection if intersection else list(a)
    return a

def get_sam_prompts(image, masks, max_indices, imgs_bboxes, config):
        
    # ------  bbox prompts cordinates relevant to ROI for SAM------
        
        bboxes = []
        relevant_crop = []
        img_with_bboxes = []

        maxindice_assembly = filter_array(max_indices[config.dataset], max_indices['background'])

        for indices in maxindice_assembly:
            bbox = masks[indices]["bbox"]
            bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            img = image.copy()

            img_with_bboxes = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 5)
            bboxes.append(bbox)

        bboxes = np.array(bboxes)
        return bboxes
 
def sam_predicton(sam, image, resize_transform, bboxes, config, mode):
        
        # ------ SAM format ------
    
        batched_input = [{
            'image': prepare_image(image, resize_transform, "cuda").to("cuda"),
            'boxes': resize_transform.apply_boxes_torch(torch.from_numpy( np.array(bboxes)), image.shape[:2]).to("cuda"),
            'original_size': image.shape[:2] 
        }]
        
        preds = sam(batched_input, multimask_output=False)
        binary_masks = torch.sigmoid(preds) > 0.5
        binary_masks = binary_masks.squeeze().cpu().numpy()

        if(len(binary_masks.shape)==2):
            return binary_masks
        # binary_masks = reduce(np.bitwise_or, binary_masks)
        binary_masks = np.bitwise_or(binary_masks[0], binary_masks[1])

        return binary_masks

def get_eval(dataset, config, prompt_mode='crops', mode="sam_clip"):

    folder_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")

   # ----- loading the models  ----- 
    
    clip_model, preprocess = clip.load("ViT-L/14", device="cuda")
    sam_checkpoint = config.sam_ckpt
    
    sam = sam_model_registry[config.model_type](checkpoint=sam_checkpoint)
    sam.to("cuda")
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    # ----- CLIP Prompts -----
    
    with open(config.clip_prompts, "r") as file:
        prompts = json.load(file)

    dice_scores = []

    mask_generator, area = hyper_params_tuning(sam)

    # ----- Inference -----
    with torch.no_grad(): 
        for idx in tqdm(range(len(dataset)), desc= f"Processing  images", unit= "sam_point"): 

            image, gt, points, bounding_boxes, contour, file_name = dataset[idx]
        
            if mode == "sam_clip":
                masks = mask_generator.generate(image)
                masksall = copy.deepcopy(masks)
                masks = [mask for mask in masks if mask["area"] < area] # area filtering based on area value from hyper-params tuning
                img_crops = get_crops(image, masks)
                if(len(img_crops)==0):
                    masks = copy.deepcopy(masksall)
                    img_crops = get_crops(image, masks, prompt_mode)
        
                max_indices, scores = retrieve_relevant_crop(img_crops, prompts, clip_model, preprocess, config)

                # ------  bbox cordinates relevant to crop ------
                bboxes = get_sam_prompts(image, masks, max_indices, img_crops, config)
                preds = sam_predicton(sam, image, resize_transform, bboxes, config, mode)

            elif mode == "sam_prompted":
                # bounding  box prompt from ground truth
                bboxes = bounding_boxes
                preds = sam_predicton(sam, image, resize_transform, bboxes, config, mode)
            print(preds.shape)
            dice_score = dc(preds, gt)
            jac = jc(preds, gt)
            hd95_score = hd95(preds, gt)
            asd_score = asd(preds, gt)


            print("dice:", dice_score, "jaccard", jac, "hd95", hd95_score, "asd", asd_score)
            dice_scores.append((dice_score, jac, hd95_score, asd_score)) 
            
            dice_scores_np = np.array(dice_scores)
            average_dice_score = np.mean(dice_scores_np[:, 0])
            average_jac = np.mean(dice_scores_np[:, 1])
            average_hc = np.mean(dice_scores_np[:, 2])
            average_asd = np.mean(dice_scores_np[:, 3])
            
            print("Average Dice Score:", average_dice_score, "jac:", average_jac, "hd95", average_hc, "asd", average_asd) 


    return average_dice_score
 


        
        








