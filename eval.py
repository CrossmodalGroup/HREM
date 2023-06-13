import os
import torch
import argparse
import logging
from lib import evaluation


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default='/mnt/data2/fzr/datasets_weights/', type=str, help='path to datasets')
    parser.add_argument('--dataset', default='coco', type=str, help='the dataset choice, coco or f30k')
    parser.add_argument('--save_results', type=int, default=0, help='if save the similarity matrix for ensemble')
    parser.add_argument('--gpu-id', type=int, default=1, help='the gpu-id for evaluation')
    
    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu_id)

    if opt.dataset == 'coco':
        weights_bases = [
            'runs/coco_test'
        ]
    else:
        weights_bases = [
            'runs/f30k_test'
        ]

    for base in weights_bases:

        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth')
        
        # Save the final similarity matrix 
        if opt.save_results:  
            save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))
        else:
            save_path = None

        if opt.dataset == 'coco':
            # Evaluate COCO 5-fold 1K
            # Evaluate COCO 5K
            evaluation.evalrank(model_path, split='testall', fold5=True, save_path=save_path, data_path=opt.data_path)
        else:
            # Evaluate Flickr30K
            evaluation.evalrank(model_path, split='test', fold5=False, save_path=save_path, data_path=opt.data_path)


if __name__ == '__main__':
    
    main()

