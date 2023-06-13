import argparse
import os


def get_argument_parser():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='/data/darkpromise/', type=str,
                        help='path to datasets')
    parser.add_argument('--dataset', default='f30k', type=str,
                        help='dataset coco or f30k')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=5e-4, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=200, type=int,
                        help='Number of steps to logger.info and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs_test/',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='runs_test/',
                        help='Path to save the model.')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    
    parser.add_argument('--vse_mean_warmup_epochs', type=int, default=1,
                        help='The number of warmup epochs using mean vse loss') 
    
    parser.add_argument('--multi_gpu', type=int, default=0, 
                        help='whether use the multi gpus for training')
    parser.add_argument('--size_augment', type=int, default=1, 
                        help='whether use the size augmention for training')
    
    parser.add_argument('--mask_repeat', type=int, default=1, 
                        help='whether mask the repeat images in the batch for vse loss')  
    parser.add_argument('--save_results', type=int, default=1,
                        help='whether save the similarity matrix for the evaluation')
    parser.add_argument('--gpu-id', type=int, default=0, 
                        help='the gpu-id for training')
    parser.add_argument('--bert_path', type=str, default='/model/darkpromise/VSE_backbone/bert-base-uncased',
                        help='the path of pretrained checkpoint')      
    parser.add_argument("--lr_schedules", default=[15, 25], type=int, nargs="+", 
                        help='epoch schedules for lr decay') 
    parser.add_argument("--decay_rate", default=0.1, type=float, 
                        help='lr decay_rate for optimizer') 

    return parser


def save_parameters(opt, save_path):

    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key], dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t'+str(sub_key)+': '+str(sub_item)
        else:
            base_str += '\n\t'+str(varx[key])
        base_str+='\n'
    
    with open(os.path.join(save_path, 'Parameters.txt'), 'w') as f:
        f.write(base_str)


if __name__ == '__main__':

    pass
    