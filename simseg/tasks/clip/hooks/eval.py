import torch

from simseg.core.hooks import Hook
from simseg.utils import ENV, logger, all_gather
from simseg.utils.collections import AttrDict
from simseg.tasks.clip.hooks.utils import RetrievalMetric, IndexedEmbInfo


class RetrievalEvalHook(Hook):

    def __init__(self, runner):
        self.retrieval = RetrievalMetric()
        self.collection_keys = ['image_embeddings', 'text_embeddings', 'image_id', 'caption_id']
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
            collection_dict[key] = value #.cuda(ENV.device)

        valid_index = collection_dict['image_id'] > -1
        collection_dict = {k: v[valid_index] for k,v in collection_dict.items()}

        collection_dict['dataset_name'] = epoch_state.get('dataset_name')

        self.calcaulate_retrieval_metrics_and_log(runner, collection_dict)

    @ENV.root_only
    def calcaulate_retrieval_metrics_and_log(self, runner, collection_dict):
        logger.info('---- Calculating retrieval metrics ... ----')
        index = collection_dict['image_id']
        image_embedding = collection_dict['image_embeddings']
        text_embedding = collection_dict['text_embeddings']

        if not runner.cfg.data.cuda_eval:
            logger.info('---- Using cpu evaluation ----')
            index = index.cpu()
            image_embedding = image_embedding.cpu()
            text_embedding = text_embedding.cpu()
        else:
            logger.info('---- Using cuda evaluation ----')

        img_emb = IndexedEmbInfo(emb_name='image',group_idx=index,emb_mat=image_embedding).unique()
        text_emb = IndexedEmbInfo(emb_name='text',group_idx=index,emb_mat=text_embedding)

        logger.info('{} validation: image emb shape: {}, text emb shape: {}'.format(collection_dict['dataset_name'], img_emb.emb_mat.shape, text_emb.emb_mat.shape))

        i2t = self.retrieval(img_emb, text_emb)
        t2i = self.retrieval(text_emb, img_emb)
        
        i2t.update(t2i)

        summary_dict = {}
        for k, v in i2t.items():
            k = k.replace('[image] to [text]', 'I2T')
            k = k.replace('[text] to [image]', 'T2I')
            k = k.replace(': ', '-')
            summary_dict[k] = v * 100.0

        summary_dict['RSUM'] = sum(list(summary_dict.values()))
        summary_dict = {'{}_{}'.format(collection_dict['dataset_name'], k): v for k, v in summary_dict.items()}

        temperature = runner.model.module.loss.temperature.detach().cpu().numpy()
        summary_dict['temperature'] = temperature

        logger.emph('-----{} summary -----'.format(collection_dict['dataset_name']))
        logger.info(summary_dict)
        if self.wandb_enable:
            runner.state.wandb_record.val_record = summary_dict
        logger.emph('-----{} summary -----'.format(collection_dict['dataset_name']))

class RetrievalLocalEvalHook(RetrievalEvalHook):
    def __init__(self, runner):
        super(RetrievalLocalEvalHook, self).__init__(runner)

    @ENV.root_only
    def after_val_epoch(self, runner, epoch_state):
        collection_dict = {}
        for key in self.collection_keys:
            value = torch.cat(epoch_state.eval[key], 0)
            collection_dict[key] = value

        valid_index = collection_dict['image_id'] > 0
        collection_dict = {k: v[valid_index] for k,v in collection_dict.items()}

        collection_dict['dataset_name'] = epoch_state.get('dataset_name')

        self.calcaulate_retrieval_metrics_and_log(runner, collection_dict)

if __name__ == "__main__":
    a = torch.rand((512, 1000)).cuda()
    b = torch.rand((512, 1000)).cuda()
    c = torch.arange(0, 512).cuda()
    retrieval = RetrievalMetric()

    img_emb = IndexedEmbInfo(emb_name='image',group_idx=c,emb_mat=a).unique()
    text_emb = IndexedEmbInfo(emb_name='text',group_idx=c,emb_mat=b)
    i2t = retrieval(img_emb, text_emb)
    t2i = retrieval(text_emb, img_emb)

    i2t.update(t2i)

    summary_dict = {}
    for k, v in i2t.items():
        k = k.replace('[image] to [text]', 'I2T')
        k = k.replace('[text] to [image]', 'T2I')
        k = k.replace(': ', '-')
        summary_dict[k] = v * 100

    summary_dict['RSUM'] = sum(list(summary_dict.values()))
    print(summary_dict)


        

