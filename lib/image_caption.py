import torch
import torch.utils.data as data
import os
import numpy as np
import random
import logging


logger = logging.getLogger(__name__)


class PrecompRegionDataset(data.Dataset):
    def __init__(self, data_path, data_split, tokenizer, opt, train):
        self.data_path = data_path    
        self.tokenizer = tokenizer
        self.opt = opt
        self.train = train

        loc = os.path.join(data_path, '{}_precomp'.format(opt.dataset))

        # Raw captions
        self.captions = []
        with open(os.path.join(loc, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())
        
        # Region features
        self.images = np.load(os.path.join(loc, '%s_ims.npy' % data_split))

        # num_captions
        self.length = len(self.captions)
        self.num_images = len(self.images)

        if self.num_images != self.length:
            # one images to five captions (train set)
            self.im_div = 5
        else:
            # one images to one captions (test set)
            self.im_div = 1

        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):

        # handle the image redundancy
        # index for captions, img_index for images
        img_index = index // self.im_div
        caption = self.captions[index]

        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)
        
        # Convert caption (string) to word ids (with Size Augmentation at training time).
        target = process_caption_bert(self.tokenizer, caption_tokens, self.train)

        image = self.images[img_index]

        if self.train and self.opt.size_augment:   
            num_features = image.shape[0]  
            rand_list = np.random.rand(num_features)
            image = image[np.where(rand_list > 0.20)]  
        
        image = torch.Tensor(image)

        return image, target, index, img_index

    def __len__(self):
        return self.length


def process_caption_bert(tokenizer, tokens, train=True):
    
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        # text -> token (basic_tokenizer.tokenize) -> sub_token (wordpiece_tokenizer.tokenize)
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        
        prob = random.random()

        # first, 20% probability use the augmenation operations
        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            # 10% randomly change token to random token from the BERT-vocab
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))                  
            # -> 40% delete the token
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    # record the index of sub_token
                    deleted_idx.append(len(output_tokens) - 1)
        # 80% probability keep the token
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    # and first and last notations for BERT model
    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']

    # Convert token to vocabulary indices, torch.float32
    target = tokenizer.convert_tokens_to_ids(output_tokens)

    # convert to the torch-tenfor 
    target = torch.Tensor(target)
    return target


def collate_fn(data):

    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions, ids, img_ids = zip(*data)

    img_ids = torch.tensor(img_ids)
    ids = torch.tensor(ids)
    
    # print(img_ids)
    repeat = len(img_ids) - len(torch.unique(img_ids))

    # Sort a data list by caption length
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    # images = torch.stack(images, 0)

    img_lengths = [len(image) for image in images]
    
    # dataset_size * max_lengths (maybe 36) * 2048 
    all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
    for i, image in enumerate(images):
        end = img_lengths[i]
        all_images[i, :end] = image[:end]

    img_lengths = torch.tensor(img_lengths)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    # count the length of each captions
    lengths = [len(cap) for cap in captions]

    # pad the redundancy with zero, in order to input BERT model as a batch
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    lengths = torch.tensor(lengths)

    # all_images: Batch_size * max_img_lengths * 2048 (the dimension of region-features)
    # targets:  Batch_size * max_cap_lengths
    return all_images, img_lengths, targets, lengths, ids, img_ids, repeat


def get_loader(data_path, data_split, tokenizer, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True):

    drop_last = True if train else False

    dset = PrecompRegionDataset(data_path, data_split, tokenizer, opt, train)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                collate_fn=collate_fn,
                                                num_workers=num_workers,
                                                drop_last=drop_last)

    return data_loader


def get_train_loader(data_path, tokenizer, batch_size, workers, opt):
    
    train_loader = get_loader(data_path, 'train', tokenizer, opt, batch_size, True, workers, train=True)
    
    return train_loader


def get_test_loader(data_path, split_name, tokenizer, batch_size, workers, opt):    
    
    test_loader = get_loader(data_path, split_name, tokenizer, opt, batch_size, False, workers, train=False)

    return test_loader


if __name__ == '__main__':

    pass
