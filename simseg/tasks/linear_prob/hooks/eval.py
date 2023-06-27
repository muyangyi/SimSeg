import torch

from simseg.core.hooks import Hook
from simseg.utils import ENV, logger, all_gather
from simseg.utils.collections import AttrDict
from simseg.tasks.linear_prob.hooks.utils import accuracy


class LinearEvalHook(Hook):

    def __init__(self, runner):
        self.collection_keys = ['prediction', 'label']
        self.wandb_enable = runner.cfg.wandb.enable

    def before_val_epoch(self, runner, epoch_state):
        epoch_state.eval = AttrDict()
        for key in self.collection_keys:
            epoch_state.eval[key] = []

    def after_val_step(self, runner, epoch_state, step_state):
        for key in self.collection_keys:
            epoch_state.eval[key].append(step_state.batch_output[key])

    def after_val_epoch(self, runner, epoch_state):
        collection_dict = {}
        for key in self.collection_keys:
            value = torch.cat(epoch_state.eval[key], 0)
            value = torch.cat(all_gather(value), 0)
            collection_dict[key] = value 

        collection_dict['dataset_name'] = epoch_state.get('dataset_name')

        self.calcaulate_retrieval_metrics_and_log(runner, collection_dict)

    @ENV.root_only
    def calcaulate_retrieval_metrics_and_log(self, runner, collection_dict):
        logger.info('---- Calculating linear probing metrics ... ----')
        prediction = collection_dict['prediction']
        label = collection_dict['label']

        logger.emph('prediction with shape {}'.format(prediction.shape))

        acc1, acc5 = accuracy(prediction, label, topk=(1, 5))

        summary_dict = {}
        summary_dict['val_acc1'] = acc1[0]
        summary_dict['val_acc5'] = acc5[0]
        summary_dict = {'{}_{}'.format(collection_dict['dataset_name'], k): v for k, v in summary_dict.items()}

        logger.emph('-----{} summary -----'.format(collection_dict['dataset_name']))
        logger.info(summary_dict)
        if self.wandb_enable:
            runner.state.wandb_record.val_record = summary_dict
        logger.emph('-----{} summary -----'.format(collection_dict['dataset_name']))


        

