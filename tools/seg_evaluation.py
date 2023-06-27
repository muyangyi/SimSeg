import cv2
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from copy import deepcopy
import pydensecrf.densecrf as dcrf
from transformers import AutoTokenizer
import torch.nn.functional as F

try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    pass
from torch.nn.parallel import DistributedDataParallel as torch_DDP

from simseg.core import init_device, cfg, update_cfg
from simseg.datasets.seg.seg_dataset import build_torch_valid_loader
from simseg.models import PIPELINE
from simseg.utils import build_from_cfg, ENV, logger, all_gather
from simseg.utils.metrics import mean_iou
from simseg.utils.prompt import openai_imagenet_template
from simseg.utils.interpolate_pe import interpolate_pos_embed

from simseg.core.hooks.checkpoint import get_dist_state_dict
from simseg.tasks.clip.config import task_cfg_init_fn, update_clip_config


def dense_crf(img, probs, n_labels=2):
    h = probs.shape[0]
    w = probs.shape[1]

    probs = np.expand_dims(probs, 0)
    probs = np.append(1 - probs, probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, n_labels)

    U = -np.log(probs + 1e-8)
    U = U.reshape((n_labels, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    U = U.astype(np.float32)
    d.setUnaryEnergy(U) 

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=40, srgb=13, rgbim=img, compat=10)

    Q = d.inference(3)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q


def zero_shot_classifier(model, classnames, make_template, tokenizer, ENV):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = make_template(classname)  # format with class
            texts = tokenizer(texts, padding='max_length', truncation=True, max_length=25)

            input_ids, attention_mask = torch.tensor(texts["input_ids"]), torch.tensor(texts["attention_mask"])
            input_ids = input_ids.to(ENV.rank)
            attention_mask = attention_mask.to(ENV.rank)

            class_embeddings = model.module.forward_text_feature(input_ids, attention_mask)
            class_embeddings = model.module.forward_text_project(class_embeddings, attention_mask)

            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.transpose(0, 1)


def evaluate_benchmark(loader, model, cfg, tokenizer, seg_categories, top_cls_num):
    image_mean = torch.tensor(cfg.transforms.normalize.mean, device=ENV.device)
    image_mean = image_mean.view(1, 3, 1, 1)
    image_norm = torch.tensor(cfg.transforms.normalize.std, device=ENV.device)
    image_norm = image_norm.view(1, 3, 1, 1)

    patch_size = 16
    num_patch = cfg.transforms.input_size // patch_size

    # extract text features for labels
    label_text_feature = zero_shot_classifier(model, seg_categories, openai_imagenet_template, tokenizer, ENV)

    count = 0
    total_intersection, total_union = 0, 0

    for batch in tqdm(loader):
        batch_dict = {}
        batch_dict['image'], batch_dict['mask_label'] = batch
        batch_dict = {k: v.cuda(ENV.device, non_blocking=True) for k,v in batch_dict.items() if k not in ['caption']}

        # image_feature includes [CLS] token
        image_feature = model.module.forward_image_feature(batch_dict['image']) # (Batch, 324, 768)
        # projector and L2 normalize
        image_feature_pooled = model.module.forward_image_project(image_feature) # (Batch, 512)
        image_feature = model.module.image_projection(image_feature) # (Batch, 324, 512)

        # visualize raw image and text
        image_raw = (((batch_dict['image'] * image_norm) + image_mean) * 255).type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        image_shape = batch_dict['mask_label'][0].shape
        mask_label = batch_dict['mask_label']

        index = 0
        image_a = Image.fromarray(image_raw[index])
        im_f_a = image_feature[index]
        im_f_a = F.normalize(im_f_a, dim=-1, p=2)
        im_avg_a = image_feature_pooled[index]

        attn_ai2ai = 1.0 * im_f_a @ im_avg_a.unsqueeze(-1) # (324, 1)
        attn_ai2ai = attn_ai2ai.reshape(num_patch, num_patch)
        attn_ai2ai = F.interpolate(attn_ai2ai.unsqueeze(0).unsqueeze(0), scale_factor=patch_size, mode="nearest")[0][0]

        i2t_scores = torch.sum(im_avg_a.unsqueeze(0) * label_text_feature, dim=1)
        scores = i2t_scores 

        topk_scores, topk_index = scores.topk(top_cls_num)
        score_mean, score_std = topk_scores.mean(), topk_scores.std()
        threshold = score_mean + 1.0 * score_std

        raw_H, raw_W = image_shape
        temp_pred = np.zeros((len(seg_categories), raw_H, raw_W))

        candidate_class_num = 5

        for i, index in enumerate(topk_index[:candidate_class_num]):
            
            if index in [0,255]:
                continue

            attn_ai2at = im_f_a @ label_text_feature[index].unsqueeze(-1) # (324, 1)
            attn_ai2at = attn_ai2at.reshape(num_patch, num_patch)

            attn_ai2at = F.interpolate(attn_ai2at.unsqueeze(0).unsqueeze(0), scale_factor=patch_size, mode="nearest")[0][0] 

            H, W = attn_ai2at.shape[-2:]
            attn_ai2at = attn_ai2at.reshape(H*W, 1)
            attn_ai2at = attn_ai2at.reshape(H, W).cpu().numpy()

            score = float(scores[index])
            if score < threshold:
                break

            min_value, max_value = attn_ai2at.min(), attn_ai2at.max()
            norm_attn = (attn_ai2at - min_value) / (max_value - min_value)

            ### use crf
            binary_mask = dense_crf(np.array(image_a).astype(np.uint8), norm_attn) * 255
            binary_mask = binary_mask.astype('uint8')

            # dilate and erode kernel
            kernel = np.ones((7, 7), dtype=np.uint8)
            final_mask = cv2.dilate(binary_mask, kernel, 5)
            final_mask = cv2.erode(final_mask, kernel, 3)

            final_mask = cv2.resize(final_mask.astype(np.uint8), dsize=(raw_W, raw_H), interpolation=cv2.INTER_NEAREST)
            temp_pred[index] = final_mask * score
        
        temp_intersection, temp_union = mean_iou(
            results=[temp_pred.argmax(0)],
            gt_seg_maps=[mask_label[0].cpu().numpy()],
            num_classes=len(seg_categories),
            ignore_index=255
        )

        total_intersection += temp_intersection
        total_union += temp_union

        count += 1

    multi_class_iou = total_intersection / total_union
    final_mean_iou = multi_class_iou[~torch.isnan(multi_class_iou)].mean()
    
    print('---------------- {} samples evaluated. ----------------'.format(count))
    logger.emph('multi class iou:', multi_class_iou)
    logger.emph('final mean iou:', final_mean_iou.data)


def parse_args():
    # Parse args with argparse tool
    parser = argparse.ArgumentParser(description='SimSeg Evaluation')
    parser.add_argument('--cfg', type=str, required=True,
                        help='experiment configure file name')
    parser.add_argument("--local_rank", type=int, default=0)  # Compatibility with torch launch.py
    parser.add_argument("--ckpt_path", type=str, default='')  
    args, cfg_overrided = parser.parse_known_args()

    # Update config from yaml and argv for override
    update_cfg(task_cfg_init_fn, args.cfg, cfg_overrided, preprocess_fn=update_clip_config)

    # Record the global config and its snapshot (for easy experiment reproduction)
    ENV.cfg = cfg
    ENV.cfg_snapshot = deepcopy(cfg)
    ENV.local_rank = args.local_rank

    return args


def main():
    # Configuration: user config updating and global config generating
    args = parse_args()

    # Initialization: set device, generate global config and inform the user library
    init_device(cfg)

    # Build model
    model = build_from_cfg(cfg.model.name, cfg, PIPELINE).to(ENV.device)

    if cfg.dist.name == 'apex':
        model = DDP(model, delay_allreduce=False)
    elif cfg.dist.name == 'torch':
        model = torch_DDP(model,
                        device_ids=[ENV.local_rank],
                        output_device=ENV.local_rank,
                        find_unused_parameters=False)
    else:
        raise NotImplementedError

    # Runner: building and running
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model_checkpoint = checkpoint['state_dict'] 

    if 'image_encoder.model.model.pos_embed' in checkpoint['state_dict']:
        pos_embed_reshaped = interpolate_pos_embed(checkpoint['state_dict']['image_encoder.model.model.pos_embed'], model.module.image_encoder.model.model)   
        checkpoint['state_dict']['image_encoder.model.model.pos_embed'] = pos_embed_reshaped
        logger.info('Interpolate PE successed.')

    model.load_state_dict(get_dist_state_dict(model_checkpoint), strict=False)
    model.eval()
    
    logger.emph(f'Loaded ckpt path: {args.ckpt_path}')

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder.tag)
    
    for name in cfg.data.valid_name:
        valid_loader = build_torch_valid_loader(cfg, name, mode='valid')

        with open(f'data/label_category/{name}.txt', 'r') as f:
            categories = f.readlines()
        seg_categories = [label.strip() for label in categories]

        top_cls_num = 30 if name == 'pascal_context' else 10

        with torch.no_grad():
            evaluate_benchmark(valid_loader, model, cfg, tokenizer, seg_categories, top_cls_num)

if __name__ == "__main__":
    main()
