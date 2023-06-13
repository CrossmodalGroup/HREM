import logging
import time
import torch
import numpy as np
from collections import OrderedDict
from transformers import BertTokenizer

from lib import image_caption
from lib.vse import VSEModel

logger = logging.getLogger(__name__)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # current values
        self.val = val
        # total values
        self.sum += val * n
        # the number of records
        self.count += n
        # average values
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=logger.info):
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    for i, data_i in enumerate(data_loader):
        
        # make sure val logger is used
        images, image_lengths, captions, lengths, ids, img_ids, repeat = data_i
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb = model.forward_emb(images, captions, img_ids, lengths, image_lengths=image_lengths)

        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))         

        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, :] = cap_emb.data.cpu().numpy().copy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Batch-Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                i, len(data_loader.dataset) // data_loader.batch_size + 1, batch_time=batch_time,
                e_log=str(model.logger)))
        del images, captions

    return img_embs, cap_embs


def eval_ensemble(results_paths, fold5=False):
    all_sims = []
    all_npts = []
    
    for sim_path in results_paths:
        results = np.load(sim_path, allow_pickle=True).tolist()
        npts = results['npts']
        sims = results['sims']
        all_npts.append(npts)
        all_sims.append(sims)

    all_npts = np.array(all_npts)  
    all_sims = np.array(all_sims)   # Equivalent to np.stack(all_sims, axis=0)

    assert np.all(all_npts == all_npts[0])

    npts = int(all_npts[0])

    # all_sims: (k_ensemble, N, 5N)
    # use the average of sim matrix from different k models.
    # Will the ensemble get better ?
    sims = all_sims.mean(axis=0)

    if not fold5:
        r, rt = i2t(npts, sims, return_ranks=True)
        ri, rti = t2i(npts, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        logger.info("rsum: %.1f" % rsum)
        logger.info("Average i2t Recall: %.1f" % ar)
        logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        logger.info("Average t2i Recall: %.1f" % ari)
        logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        npts = npts // 5
        results = []
        all_sims = sims.copy()
        for i in range(5):
            sims = all_sims[i * npts: (i + 1) * npts, i * npts * 5: (i + 1) * npts * 5]
            r, rt0 = i2t(npts, sims, return_ranks=True)
            logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(npts, sims, return_ranks=True)
            logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            logger.info("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]
        logger.info("-----------------------------------")
        logger.info("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        logger.info("rsum: %.1f" % (mean_metrics[12]))
        logger.info("Average i2t Recall: %.1f" % mean_metrics[10])
        logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[:5])
        logger.info("Average t2i Recall: %.1f" % mean_metrics[11])
        logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[5:10])


def evalrank(model_path, opt=None, tokenizer=None, model=None, split='dev', fold5=False, save_path=None, data_path=None):

    # load model and options
    checkpoint = torch.load(model_path, map_location='cuda')
    
    # (1) construct opt
    if opt is None:
        opt = checkpoint['opt']

    # (2) construct tokenizer
    if tokenizer is None:
        # tokenizer = BertTokenizer.from_pretrained(opt.bert_path)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # (3) construct model
    if model is None:
        model = VSEModel(opt, eval=True)
        # print(model.img_enc)
        # print(model.txt_enc)
        if opt.multi_gpu:
            model.make_data_parallel()

    # (4) load model checkpoint
    model.load_state_dict(checkpoint['model'])
    model.val_start()

    # (5) load the test-set
    logger.info('Loading dataset.')
    if data_path is None:
        data_path = opt.data_path
    data_loader = image_caption.get_test_loader(data_path, split, tokenizer, opt.batch_size, opt.workers, opt)

    # (6) compute the embeddings
    logger.info('Computing embeddings.')
    with torch.no_grad():
        img_embs, cap_embs = encode_data(model, data_loader)

    # one image to five captions, since have repetitive images
    logger.info('Images: %d, Captions: %d' % (img_embs.shape[0] / 5, cap_embs.shape[0]))

    # (7) compute the similarity
    # for f30k, imgs 1000, captions 5000; 
    # for coco, imgs 5000, captions 25000
    logger.info('Computing results.')

    img_embs = img_embs[::5]

    sims = compute_sim(img_embs, cap_embs)

    logger.info('Start evaluation.')

    # npts = the number of images
    npts = img_embs.shape[0]

    if save_path is not None:
        np.save(save_path, {'npts': npts, 'sims': sims})
        logger.info('Save the similarity into {}'.format(save_path))

    r, rt = i2t(npts, sims, return_ranks=True)
    ri, rti = t2i(npts, sims, return_ranks=True)

    # r[0] -> R@1, r[1] -> R@5, r[2] -> R@10
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3

    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    logger.info("rsum: %.1f" % rsum)
    logger.info("Image to text (R@1, R@5, R@10): %.1f %.1f %.1f" % r[:3])
    logger.info("Text to image (R@1, R@5, R@10): %.1f %.1f %.1f" % ri[:3])
    
    # 5-fold cross-validation
    # only for MSCOCO
    if fold5:
        logger.info('Start evaluation on 5-fold 1K for MSCOCO.')
        results = []
        sims_all = sims
        for i in range(5):
            sims = sims_all[i * 1000:(i + 1) * 1000, i * 5000:(i + 1) * 5000]

            npts = sims.shape[0]
            r, rt0 = i2t(npts, sims, return_ranks=True)
            ri, rti0 = t2i(npts, sims, return_ranks=True)

            logger.info("Image to text: %.1f, %.1f, %.1f" % r[:3])
            logger.info("Text to image: %.1f, %.1f, %.1f" % ri[:3])

            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        logger.info("-----------------------------------")
        logger.info("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        logger.info("rsum: %.1f" % (mean_metrics[12]))
        logger.info("Image to text (R@1, R@5, R@10): %.1f %.1f %.1f" % mean_metrics[:3])
        logger.info("Text to image (R@1, R@5, R@10): %.1f %.1f %.1f" % mean_metrics[5:8])


def compute_sim(images, captions):
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities


def i2t(npts, sims, return_ranks=False, mode='coco', special_list=None):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths

    sims: (N, 5N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        if mode == 'coco':
            rank = 1e20
            for i in range(5 * index, 5 * index + 5, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            top1[index] = inds[0]
        else:
            rank = np.where(inds == index)[0][0]
            ranks[index] = rank
            top1[index] = inds[0]

    # Compute metrics
    if special_list is not None:
        ranks = ranks[np.array(special_list)]
        print('num of special_index is:', len(ranks))

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(npts, sims, return_ranks=False, mode='coco', special_list=None):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    # npts = images.shape[0]

    if mode == 'coco':
        ranks = np.zeros(5 * npts)
        top1 = np.zeros(5 * npts)
    else:
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        if mode == 'coco':
            for i in range(5):
                inds = np.argsort(sims[5 * index + i])[::-1]
                ranks[5 * index + i] = np.where(inds == index)[0][0]
                top1[5 * index + i] = inds[0]
        else:
            inds = np.argsort(sims[index])[::-1]
            ranks[index] = np.where(inds == index)[0][0]
            top1[index] = inds[0]

    if special_list is not None:
        ranks = ranks[np.array(special_list)]
        print('num of special_index is:', len(ranks))  
         
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


if __name__ == '__main__':

    pass
    