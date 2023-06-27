import os
from simseg.utils import logger
import yaml
from copy import deepcopy

from simseg.utils.collections import AttrDict


def task_cfg_init_fn(cfg : AttrDict):
    # Define basic configurations

    cfg.runner.name = 'clip'
    # how many step to log
    cfg.runner.log_interval = 1 
    # how many epoch to valid
    cfg.runner.val_interval = 1 
    # how many steps to valid, if set, cfg.runner.val_interval will be ignored
    cfg.runner.val_interval_steps = -1  #是否按step来做validation，若设置为>0, 则每val_interval_steps次step做一次evaluate
    cfg.runner.stable_random = "none"

    cfg.wandb = AttrDict()
    cfg.wandb.enable = False # 是否打开wandb
    cfg.wandb.project = 'f30k' # wandb的project
    cfg.wandb.entity = 'zeromml' # wandb的entity
    cfg.wandb.train_record_keys = ['loss', 'lr', 'train_acc1', 'train_acc5'] # 需要往wandb里打的参数的key，这些参数需要能从batch_outputs中取到，一般不需改

    #step checkpoint save and resume
    cfg.ckpt.dir = './output' #实验model和log存放的根目录
    cfg.ckpt.step_interval = 2000 
    cfg.ckpt.filename = 'step_checkpoint.pth' 
    cfg.ckpt.external_resume = None
    cfg.ckpt.only_load_image_encoder = False
    cfg.ckpt.only_load_text_encoder = False
    cfg.ckpt.soft_resume = False
    cfg.ckpt.auto_resume = True

    #log interval
    cfg.log.interval_train = 1
    cfg.log.interval_val = 1

    # Distributed training configurations
    cfg.dist.name = 'torch'
    cfg.dist.param = dict()
    cfg.dist.fp16 = True #for torch dist 

    # Optimizing configurations
    cfg.optim.name = 'torch.optim.AdamW'
    cfg.optim.param = dict(betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1)
    cfg.optim.grad_clip = dict()

    # optimizer lr
    cfg.optim.lr.name = 'cosine_schedule_with_warmup' # name of lr scheduler
    cfg.optim.lr.init = 1e-4
    cfg.optim.lr.warmup_proportion = 0.025
    cfg.optim.lr.param = dict(num_cycles=0.5)

    # ----- DATASET BUILDER -----
    cfg.data.exp_name = 'test' # log和model存放位置，和openai code中的name一致, 这个name也同时是wandb和parquet数据流的名字
    cfg.data.name = 'parquet' 
    cfg.data.train_type = 'sequential' # sequential & shuffle & batch_shuffle

    cfg.data.train_name = ['imagenet1k'] # 可到zerommml_datasets.py中查询有哪些可用 
    cfg.data.valid_name = ['imagenet1k'] # 可到zerommml_datasets.py中查询有哪些可用

    cfg.data.data_path = './data/' 

    cfg.data.batch_size = 128 # 注意此处的batch_size是训练时所有gpu上的batch_size，每张卡上的batch_size为 cfg.data.batch_size//ENV.size
    cfg.data.batch_size_train = 128 # grad accumulation
    cfg.data.batch_size_val = 256 # 同上，validation时所有卡上的batch_size之和
    cfg.data.num_workers = 8

    cfg.data.enable_valid = True # 是否需要做evaluate # 是否做batchsize warmup
    cfg.data.single_eval = True # 是否做单卡测试，设置成False则做分布式测试
    cfg.data.cuda_eval = True # retrieval矩阵计算时是否要用cuda，注意到潮汐集群上时需要设置成False，不然容易爆显存，实测cpu不会慢多少(1s vs 10s)
    
    # ----- TRAMSFORM BUILDER -----
    cfg.transforms = AttrDict()

    cfg.transforms.input_size = 224 # declare for vit

    cfg.transforms.train_transforms =  ["resize"]
    cfg.transforms.valid_transforms = ["resize"]

    cfg.transforms.resize = AttrDict()
    cfg.transforms.resize.size = 224
    cfg.transforms.resize_bicubic = AttrDict()
    cfg.transforms.resize_bicubic.size = 224

    cfg.transforms.normalize = AttrDict()
    cfg.transforms.normalize.mean = [0.485, 0.456, 0.406]
    cfg.transforms.normalize.std = [0.229, 0.224, 0.225]

    cfg.transforms.random_crop = AttrDict()
    cfg.transforms.random_crop.size = 224

    cfg.transforms.center_crop = AttrDict()
    cfg.transforms.center_crop.size = 224

    cfg.transforms.random_resize_crop = AttrDict()
    cfg.transforms.random_resize_crop.size = 224
    cfg.transforms.random_resize_crop.scale = [0.6, 1.0]

    cfg.transforms.random_augment = AttrDict()
    cfg.transforms.random_augment.N = 2
    cfg.transforms.random_augment.M = 7

    cfg.transforms.mixup = 0.8
    cfg.transforms.cutmix = 1.0
    cfg.transforms.cutmix_minmax = None
    cfg.transforms.mixup_prob = 1.0
    cfg.transforms.mixup_switch_prob = 0.5
    cfg.transforms.mixup_mode = 'batch'

    cfg.transforms.random_erasing = AttrDict()
    cfg.transforms.random_erasing.reprob = 0.
    cfg.transforms.random_erasing.remode = 'pixel'
    cfg.transforms.random_erasing.recount = 1

    cfg.transforms.color_jitter = 0.4

    # ----- MODEL BUILDER -----
    cfg.model.name = 'clip'
    cfg.model.pretrain_prefix_change_list = []
    
    cfg.model.max_length = 25
    cfg.model.syncbn = True
    cfg.model.interpolate_pos_embed = False
    cfg.model.freeze_cnn_bn = False

    cfg.model.image_encoder = AttrDict()
    cfg.model.image_encoder.name = 'timm_modelzoo'
    cfg.model.image_encoder.tag = 'vit_base_patch16_224_in21k'
    cfg.model.image_encoder.embedding_dim = 768
    cfg.model.image_encoder.pretrained = True
    cfg.model.image_encoder.trainable = True
    cfg.model.image_encoder.vit = AttrDict()
    cfg.model.image_encoder.vit.only_cls_token = True

    cfg.model.text_encoder = AttrDict()
    cfg.model.text_encoder.name = 'huggingface_modelzoo'
    cfg.model.text_encoder.tag = "bert-base-uncased"
    cfg.model.text_encoder.embedding_dim = 768
    cfg.model.text_encoder.pretrained = True
    cfg.model.text_encoder.trainable = True
    cfg.model.text_encoder.target_token_idx = 0
    cfg.model.text_encoder.only_cls_token = False
    
    cfg.model.classifier = AttrDict()
    cfg.model.classifier.num_classes = 512

    cfg.model.pool = AttrDict()
    cfg.model.pool.name = 'identity' #identity gpo topkpooling1d betapo


    # ----- LOSS BUILDER -----
    cfg.loss = AttrDict()
    cfg.loss.name = 'NCE'
    cfg.loss.global_reduce = True
    cfg.loss.group_size = -1
    cfg.loss.smoothing = 0.0

    cfg.loss.extra_losses = []

    cfg.loss.nce_loss = AttrDict()
    cfg.loss.nce_loss.gather_backward = False

    cfg.loss.mixup = AttrDict()
    cfg.loss.mixup.beta = 0.1

    cfg.loss.temperature = AttrDict()
    cfg.loss.temperature.name = 'constant' #constant paramater log_parameter
    cfg.loss.temperature.value = 0.02

    cfg.loss.triplet_loss = AttrDict()
    cfg.loss.triplet_loss.reduce_mode = 'max'
    cfg.loss.triplet_loss.margin = 0.2
    

def update_clip_config(cfg : AttrDict):
    cfg.ckpt.dir = os.path.join(cfg.ckpt.dir, cfg.data.exp_name)
    
    if isinstance(cfg.data.batch_size, list):
        cfg.data.batch_size = cfg.data.batch_size[0]

    if isinstance(cfg.data.batch_size_val, list):
        cfg.data.batch_size_val = cfg.data.batch_size_val[0]
