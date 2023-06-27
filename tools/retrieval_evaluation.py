import argparse
import torch

from copy import deepcopy
from pprint import pprint

try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    pass
from torch.nn.parallel import DistributedDataParallel as torch_DDP

from simseg.core import init_device, cfg, update_cfg
from simseg.datasets.clip.clip_dataset import build_torch_valid_loader
from simseg.models import PIPELINE
from simseg.utils import build_from_cfg, ENV, logger, all_gather

from simseg.core.hooks.checkpoint import get_dist_state_dict

from simseg.tasks.clip.hooks.utils import RetrievalMetric, IndexedEmbInfo
from simseg.tasks.clip.config import task_cfg_init_fn, update_clip_config


@ENV.root_only
def calcaulate_retrieval_metrics_and_log(collection_dict, cuda_eval = True):
    retrieval = RetrievalMetric()
    index = collection_dict['image_id'] if collection_dict['dataset_name'] != 'imagenet' else collection_dict['caption_id']
    image_embedding = collection_dict['image_embeddings']
    text_embedding = collection_dict['text_embeddings']

    if not cuda_eval:
        index = index.cpu()
        image_embedding = image_embedding.cpu()
        text_embedding = text_embedding.cpu()

    if collection_dict["dataset_name"] != 'imagenet':
        img_emb = IndexedEmbInfo(emb_name='image',group_idx=index,emb_mat=image_embedding).unique()
        text_emb = IndexedEmbInfo(emb_name='text',group_idx=index,emb_mat=text_embedding)
    else:
        img_emb = IndexedEmbInfo(emb_name='image',group_idx=index,emb_mat=image_embedding)
        text_emb = IndexedEmbInfo(emb_name='text',group_idx=index,emb_mat=text_embedding).unique()

    logger.info('{} validation: image emb shape: {}, text emb shape: {}'.format(collection_dict['dataset_name'], img_emb.emb_mat.shape, text_emb.emb_mat.shape))

    i2t = retrieval(img_emb, text_emb)
    t2i = retrieval(text_emb, img_emb)
    
    i2t.update(t2i)

    summary_dict = {}
    for k, v in i2t.items():
        k = k.replace('[image] to [text]', 'I2T')
        k = k.replace('[text] to [image]', 'T2I')
        k = k.replace(': ', '-')
        summary_dict[k] = v * 100.0

    summary_dict['RSUM'] = sum(list(summary_dict.values()))
    summary_dict = {'{}_{}'.format(collection_dict['dataset_name'], k): v for k, v in summary_dict.items()}

    logger.emph('-------------- {} Evaluation --------------'.format(collection_dict['dataset_name']))
    pprint(summary_dict)
    logger.emph('-------------- {} Evaluation --------------\n'.format(collection_dict['dataset_name']))

def evaluate_benchmark(loader, model, name):
    collection_keys = ['image_embeddings', 'text_embeddings', 'image_id', 'caption_id']

    epoch_state = {}
    for key in collection_keys:
        epoch_state[key] = []

    for batch in loader:
        batch_dict = {}
        batch_dict['image'], batch_dict['input_ids'], batch_dict['attention_mask'], \
                batch_dict['caption'], batch_dict['image_id'], batch_dict['caption_id'] = batch
        batch_dict = {k: v.cuda(ENV.device, non_blocking=True) for k,v in batch_dict.items() if k not in ['caption']}
        image_embeddings, text_embeddings = model(batch_dict, embeddings='all')

        output = {'image_embeddings': image_embeddings,
                    'text_embeddings': text_embeddings,
                    'image_id': batch_dict['image_id'],
                    'caption_id': batch_dict['caption_id']}

        for key in collection_keys:
            epoch_state[key].append(output[key])

    collection_dict = {}
    for key in collection_keys:
        value = torch.cat(epoch_state[key], 0)
        value = torch.cat(all_gather(value), 0)
        collection_dict[key] = value

    valid_index = collection_dict['image_id'] > -1
    collection_dict = {k: v[valid_index] for k,v in collection_dict.items()}

    collection_dict['dataset_name'] = name
    calcaulate_retrieval_metrics_and_log(collection_dict)

def parse_args():
    # Parse args with argparse tool
    parser = argparse.ArgumentParser(description='simseg Evaluation')
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
    model.load_state_dict(get_dist_state_dict(model_checkpoint), strict=False)
    model.eval()

    logger.emph(f'Loaded ckpt path: {args.ckpt_path}')

    for name in cfg.data.valid_name:
        valid_loader = build_torch_valid_loader(cfg, name, mode='valid')
        with torch.no_grad():
            evaluate_benchmark(valid_loader, model, name)

if __name__ == "__main__":
    main()
