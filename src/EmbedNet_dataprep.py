"""
This code is used for preparing the Triplet dataset for EmbedNet
"""
# Standard imports
import os
import pdb
import pickle
import argparse

# Third party imports
import numpy as np
from tqdm import tqdm
import Levenshtein as lev

parser = argparse.ArgumentParser(description='Data Preperation for deep word recognition')

# Arguments for text and embeddings path
parser.add_argument('--image_embeds', default='/ssd_scratch/cvit/sid/embeddings/topk_preds_1500featsImg.npy', help='Path to the image embeddings')
parser.add_argument('--text_embeds', default='/ssd_scratch/cvit/sid/embeddings/topk_preds_1500featsSynth.npy', help='Path to the text embeddigns')
parser.add_argument('--image_info', default='/ssd_scratch/cvit/sid/image_embed_top_k_1500.txt', help='Path to the file containing word image information')
parser.add_argument('--text_info', default='/ssd_scratch/cvit/sid/top_preds_embeds_with_confidence_1500.txt', help='Path to the file containing text output information')

# model path and name arguments
parser.add_argument('--base_path', default='/ssd_scratch/cvit/sid/', help='Path to the base directory where the training and testing data is stored')
parser.add_argument('--file_name', default='EmbedNet_data', help='Name of the data file')

# Training and testing split flag
parser.add_argument('--train_percent', default=0.8, type=float, help='Percent of train data')
parser.add_argument('--semi_hard', default=False, action='store_true', help='If True semi-hard examples will also be included')
parser.add_argument('--save', default=False, action='store_true', help='If true data will be saved in ssd_scratch')

args = parser.parse_args()
print(args)

data_path = args.base_path + args.file_name

print('[INFO] Loading embeddings and text files...')
image_embeds = np.load(args.image_embeds)
try:
    topk_embeds = np.load(args.text_embeds)
except Exception as e:
    print('[INFO] Loading text embeddings in memmap mode...')
    topk_embeds = np.memmap(args.text_embeds, dtype=np.float32, mode='r', shape=(2109500, 2048))

with open(args.image_info, 'r') as image_file:
    image_info = image_file.readlines()
image_info = [item.split()[1] for item in image_info]

with open(args.text_info, 'r') as text_file:
    text_info = text_file.readlines()
text_info = [item.split()[1] for item in text_info]

# This piece is for handling text files with more data as compared to the numpy files
# text_info = text_info[:topk_embeds.shape[0]]
# image_info = image_info[:image_embeds.shape[0]]

# # Getting count of number of words in training set
split_count = int(args.train_percent * len(image_info))
image_info = image_info[:split_count]
text_info = text_info[:split_count*20]
image_embeds = image_embeds[:split_count]
topk_embeds = topk_embeds[:split_count*20]

text_dict = dict()
embeds_dict = dict()
ko = 0
k = 20
"""Text Dictionary is in the form
{'word':[([top_20_preds],[lev_dist]), (..., ...), ...], ...}
Embedding dictionary is in the form
{'word': {'image_embeds': [all image_embeds occurances], 'text_embeds': [[top_20_text_embeds], [top_20_text_embeds], ...]}, ...}
"""
for word in tqdm(image_info, desc='[INFO] Text Dict'):
    if word not in text_dict.keys():
        text_dict[word] = [(text_info[ko: k], [lev.distance(word, item) for item in text_info[ko: k]])]
    else:
        text_dict[word].append((text_info[ko: k], [lev.distance(word, item) for item in text_info[ko: k]]))
    ko = k
    k += 20

ko = 0
k = 20
for count, image_embed in enumerate(tqdm(image_embeds, desc='[INFO] Embeds Dict')):
    word = image_info[count]
    if word not in embeds_dict.keys():
        embeds_dict[word] = {'image_embeds': [image_embed], 'text_embeds': [topk_embeds[ko: k]]}
    else:
        embeds_dict[word]['image_embeds'].append(image_embed)
        embeds_dict[word]['text_embeds'].append(topk_embeds[ko: k])
    ko = k
    k += 20

final_list = list()
for word in tqdm(text_dict.keys(), desc='[INFO] Data Prep'):
    predictions = text_dict[word]
    image_embeddings, text_embeddings = embeds_dict[word]['image_embeds'], np.array(embeds_dict[word]['text_embeds'])
    for instance_count, single_instance in enumerate(predictions):
        top20_preds, top20_edit_dist = single_instance[0], single_instance[1]
        instance_text_embeds = text_embeddings[instance_count]
        anchor = image_embeddings[instance_count]
        positive = None
        negative_list = list()
        if args.semi_hard:
            semi_negative_list = list()
        for count, pred in enumerate(top20_preds):
            if word == pred:
                positive = instance_text_embeds[count]
            else:
                if not args.semi_hard:
                    negative_list.append(instance_text_embeds[count])
            if args.semi_hard:            
                condition = True
                no_inf = 1000
                while condition and no_inf != 0:
                    random_num = np.random.randint(low=1, high=len(topk_embeds))
                    random_embedding = topk_embeds[random_num]
                    if np.linalg.norm(anchor - random_embedding) > 0.4:
                        condition = False
                    no_inf -= 1
                semi_negative_list.append(random_embedding)
        if args.semi_hard:
            for semi_hard_neg_embed in semi_negative_list:
                if positive is None:
                    pass
                else:
                    final_list.append({'anchor': anchor, 'positive': positive, 'negative': np.array(semi_hard_neg_embed)})
        else:
            for negative_embeds in negative_list:
                if positive is None:    # There are a few cases when the OCR even fails to predics in Top20 predicitons
                    pass
                else:
                    final_list.append({'anchor': anchor, 'positive': positive, 'negative': np.array(negative_embeds)})

def check(final_list):
    positive_distance = list()
    negative_distance = list()
    for sample in tqdm(final_list, desc='[INFO] Checking'):
        anchor = sample['anchor']
        positive = sample['positive']
        negative = sample['negative']
        try:
            positive_distance.append(np.linalg.norm(anchor - positive))
            negative_distance.append(np.linalg.norm(anchor - negative))
        except Exception as e:
            print(e)
            pdb.set_trace()
    print('[INFO] Mean distance of anchors with positive pairs is {} Max {} Min {}.\n[INFO] Mean distance of anchor with negative pairs is {} Max {} Min {}.'.format(np.mean(positive_distance), np.max(positive_distance), np.min(positive_distance), np.mean(negative_distance), np.max(negative_distance), np.min(negative_distance)))

check(final_list)
if args.save:
    pickle.dump(final_list, open(data_path, 'wb'))
    print('[INFO] Total number of triples generated: {}\n[INFO] Pickle file saved at {}'.format(len(final_list), data_path))
