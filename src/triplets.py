"""
This piece of code is used for generating triplets on the fly while training
the embednet
"""
import pdb
import torch
import numpy as np
from tqdm import tqdm
import Levenshtein as lev
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances


class Triplets():
    def __init__(
            self,
            topk_info,
            image_info,
            topk_embeds,
            image_embeds,
            train_percent,
            margin,
            verbose=False):
        self.train_percent = train_percent
        self.split_count = int(self.train_percent * len(image_info))
        self.topk_info = topk_info[:self.split_count*20]
        self.topk_embeds = np.array(topk_embeds[:self.split_count*20])
        self.image_info = image_info[:self.split_count]
        self.image_embeds = image_embeds[:self.split_count]
        self.verbose = verbose
        self.margin = margin
        if self.verbose:
            print('[INFO] Total images are {}; total topk predictions '
                  'are {}.'.format(len(self.image_info), len(self.topk_info)))

        self.text_dict = dict()
        self.embeds_dict = dict()
        self.preprocess()

    def preprocess(self, embednet=False):
        """Text Dictionary is in the form
        {'word':[([top_20_preds],[lev_dist]), (..., ...), ...], ...}
        Embedding dictionary is in the form
        {'word': {'image_embeds': [all image_embeds occurances],
        'text_embeds': [[top_20_text_embeds], [top_20_text_embeds], ...]},
        ...}
        """
        ko = 0
        k = 20
        if not embednet:
            for word in tqdm(
                    self.image_info,
                    desc='[INFO] Text Dict',
                    disable=not self.verbose):
                if word not in self.text_dict.keys():
                    self.text_dict[word] = [
                        (self.topk_info[ko: k],
                            [lev.distance(word, item) for item in
                                self.topk_info[ko: k]])]
                else:
                    self.text_dict[word].append((
                        self.topk_info[ko: k],
                        [lev.distance(word, item)
                            for item in self.topk_info[ko: k]]))
                ko = k
                k += 20

        ko = 0
        k = 20
        if embednet:
            self.embeds_dict_embednet = dict()
            for count, image_embed in enumerate(tqdm(
                    self.image_embeds_embednet,
                    desc='[INFO] embednetEmbeds Dict',
                    disable=not self.verbose)):
                word = self.image_info[count]
                if word not in self.embeds_dict_embednet.keys():
                    self.embeds_dict_embednet[word] = {
                        'image_embeds': [image_embed],
                        'text_embeds': [self.topk_embeds_embednet[ko: k]]}
                else:
                    self.embeds_dict_embednet[word]['image_embeds'].append(
                        image_embed)
                    self.embeds_dict_embednet[word]['text_embeds'].append(
                        self.topk_embeds_embednet[ko: k])
                ko = k
                k += 20
        else:
            for count, image_embed in enumerate(tqdm(
                    self.image_embeds,
                    desc='[INFO] Embeds Dict',
                    disable=not self.verbose)):
                word = self.image_info[count]
                if word not in self.embeds_dict.keys():
                    self.embeds_dict[word] = {
                        'image_embeds': [image_embed],
                        'text_embeds': [self.topk_embeds[ko: k]]}
                else:
                    self.embeds_dict[word]['image_embeds'].append(
                        image_embed)
                    self.embeds_dict[word]['text_embeds'].append(
                        self.topk_embeds[ko: k])
                ko = k
                k += 20

    def initial_list(self):
        """
        This method generates an initial list for the embednet experiments
        """
        self.initial_list_ = list()
        if self.verbose:
            hard_neg = 0
            semi_hard_neg = 0
            easy_neg = 0
        for word in tqdm(
                self.text_dict.keys(),
                desc='[INFO] Data Prep',
                disable=not self.verbose):
            # Getting predictions, image's embedding and OCR prediction
            # text's embeddings for the word under consideration
            predictions = self.text_dict[word]
            image_embedding, text_embeddings = \
                self.embeds_dict[word]['image_embeds'],\
                self.embeds_dict[word]['text_embeds']
            # Looping over all the predictions
            for instance_count, single_instance in enumerate(predictions):
                # Getting OCR top20 predictions and edit distance wrt.
                # 1 particular instance of the word under consideration
                top20_preds = single_instance[0]
                # Getting instance text embeddings and anchor's embedding
                instance_text_embeds, anchor = \
                    text_embeddings[instance_count],\
                    image_embedding[instance_count]
                # Initialising hard, semi-hard and easy list
                hard_negative_list, semi_hard_neg_list, easy_neg_list\
                    = list(), list(), list()
                # Getting euclidean distance between anchor and all
                # the text embeddings
                top20_euclidean_distance = pairwise_distances(
                    anchor.reshape(
                        1,
                        anchor.shape[0]
                        ), instance_text_embeds)[0]
                # Boolean list with value = True when OCR's prediction
                # and word under considearion are same (correct OCR prediction)
                positive_detection = \
                    [orig == item for orig, item in zip(
                        [word] * 20,
                        top20_preds)]
                # If none of the OCR's predictions are correct,
                # then we don't need to proceed further as there will
                # be no +ive examples to process
                if True not in positive_detection:
                    continue
                # Getting euclidean distance between positive
                # prediciton's embedding and anchor
                anchor_positive_distance = top20_euclidean_distance[
                    np.where(positive_detection)[0][0]]
                # Getting positive prediction's embeddings
                positive = instance_text_embeds[
                    np.where(positive_detection)[0][0]]
                # Creating hard, semi-hard and easy lists based on
                # https://www.notion.so/06-06-20-ce28d08e3eac4219b5a72671f0c5561e
                for count, dist in enumerate(top20_euclidean_distance):
                    if dist < anchor_positive_distance:
                        hard_negative_list.append(instance_text_embeds[count])
                        if self.verbose:
                            hard_neg += 1
                    elif anchor_positive_distance < dist\
                            < anchor_positive_distance + self.margin:
                        semi_hard_neg_list.append(instance_text_embeds[count])
                        if self.verbose:
                            semi_hard_neg += 1
                    else:
                        easy_neg_list.append(instance_text_embeds[count])
                        if self.verbose:
                            easy_neg += 1
                # Merging hard and semi-hard negative list (Reason for
                # creating them differently is for future code, we
                # might need to use these lists independently)
                semi_hard_neg_list.extend(hard_negative_list)
                # Adding the data to the final list
                for neg in semi_hard_neg_list:
                    self.initial_list_.append({
                        'anchor': anchor,
                        'positive': positive,
                        'negative': neg})
        if self.verbose:
            print('[INFO] Number of hard negatives {}, semi-hard '
                  'negatives {}, easy negatives {}'.format(
                    hard_neg,
                    semi_hard_neg,
                    easy_neg))
        return self.initial_list_

    def embednet_embeds(self, model, out_shape):
        self.updated_data = list()
        BATCH = 1024
        # for word in tqdm(self.text_dict.keys(), desc='[INFO]
        # Data Prep', disable=not self.verbose):
        self.topk_embeds_embednet = np.zeros((
            self.topk_embeds.shape[0],
            out_shape))
        self.image_embeds_embednet = np.zeros((
            self.image_embeds.shape[0],
            out_shape))
        topkloader = DataLoader(
            dataset=self.topk_embeds,
            num_workers=1,
            batch_size=BATCH,
            shuffle=False,
            pin_memory=True)
        imagloader = DataLoader(
            dataset=self.image_embeds,
            num_workers=1,
            batch_size=BATCH,
            shuffle=False,
            pin_memory=True)
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda().double()
        ko = 0
        k = BATCH
        for count, embedding in enumerate(
                tqdm(
                    topkloader,
                    desc='[INFO] TopKEmbeds pass',
                    disable=not self.verbose)):
            if torch.cuda.is_available():
                output = model(embedding.cuda().double())
            else:
                output = model(embedding.double())
            self.topk_embeds_embednet[ko:k, :] = output.cpu().detach().numpy()
            ko += BATCH
            k += BATCH
        ko = 0
        k = BATCH
        for count, embedding in enumerate(
                tqdm(
                    imagloader,
                    desc='[INFO] ImageEmbeds pass',
                    disable=not self.verbose)):
            if torch.cuda.is_available():
                output = model(embedding.cuda().double())
            else:
                output = model(embedding.double())
            self.image_embeds_embednet[ko:k, :] = output.cpu().detach().numpy()
            ko += BATCH
            k += BATCH
        self.preprocess(embednet=True)
        if self.verbose:
            hard_neg = 0
            semi_hard_neg = 0
            easy_neg = 0
        for word in tqdm(
                self.text_dict.keys(),
                desc='[INFO] Data Prep',
                disable=not self.verbose):
            predictions = self.text_dict[word]
            image_embedding, text_embeddings =\
                self.embeds_dict_embednet[word]['image_embeds'],\
                self.embeds_dict_embednet[word]['text_embeds']
            image_embedding_orig, text_embeddings_orig = \
                self.embeds_dict[word]['image_embeds'],\
                self.embeds_dict[word]['text_embeds']
            # Looping over all the predictions
            for instance_count, single_instance in enumerate(predictions):
                # Getting OCR top20 predictions and edit distance wrt.
                # 1 particular instance of the word under consideration
                top20_preds = single_instance[0]
                # Getting instance text embeddings and anchor's embedding
                instance_text_embeds, instance_text_embeds_orig, anchor,\
                    anchor_orig = text_embeddings[instance_count],\
                    text_embeddings_orig[instance_count],\
                    image_embedding[instance_count],\
                    image_embedding_orig[instance_count]
                # Initialising hard, semi-hard and easy list
                hard_negative_list, semi_hard_neg_list, \
                    easy_neg_list = list(), list(), list()
                # Getting euclidean distance between anchor
                # and all the text embeddings
                top20_euclidean_distance = pairwise_distances(
                    anchor.reshape(1, anchor.shape[0]),
                    instance_text_embeds)[0]
                # Boolean list with value = True when OCR's prediction
                # and word under considearion are same
                # (correct OCR prediction)
                positive_detection = [
                    orig == item for orig,
                    item in zip([word] * 20, top20_preds)]
                # Getting euclidean distance between positive
                # prediciton's embedding and anchor
                if True not in positive_detection:
                    continue
                # Getting euclidean distance between positive
                # prediciton's embedding and anchor
                anchor_positive_distance = top20_euclidean_distance[
                    np.where(positive_detection)[0][0]]
                # Getting positive prediction's embeddings
                positive = instance_text_embeds_orig[
                    np.where(positive_detection)[0][0]]
                # Creating hard, semi-hard and easy lists based on
                # https://www.notion.so/06-06-20-ce28d08e3eac4219b5a72671f0c5561e
                for count, dist in enumerate(top20_euclidean_distance):
                    if dist < anchor_positive_distance:
                        hard_negative_list.append(
                            instance_text_embeds_orig[count])
                        if self.verbose:
                            hard_neg += 1
                    elif anchor_positive_distance < \
                            dist < anchor_positive_distance + self.margin:
                        semi_hard_neg_list.append(
                            instance_text_embeds_orig[count])
                        if self.verbose:
                            semi_hard_neg += 1
                    else:
                        easy_neg_list.append(instance_text_embeds_orig[count])
                        if self.verbose:
                            easy_neg += 1
                semi_hard_neg_list.extend(hard_negative_list)
                # Merging hard and semi-hard negative list (Reason for
                # creating them differently is for future code, we
                # might need to use these lists independently)
                for neg in semi_hard_neg_list:
                    self.updated_data.append({
                        'anchor': anchor_orig,
                        'positive': positive,
                        'negative': neg})
        if self.verbose:
            print('[INFO] Number of hard negatives {}, semi-hard '
                  'negatives {}, easy negatives {}'.format(
                    hard_neg,
                    semi_hard_neg,
                    easy_neg))
        return self.updated_data, hard_neg

    def check(self, embednet=False):
        """
        This method is used for checking the data generated,
        it prints the mean stats for the list
        """
        positive_distance = list()
        negative_distance = list()
        if not embednet:
            to_check = self.initial_list_
        else:
            to_check = self.updated_data
        for sample in tqdm(
                to_check,
                desc='[INFO] Checking',
                disable=not self.verbose):
            anchor = sample['anchor']
            positive = sample['positive']
            negative = sample['negative']
            try:
                positive_distance.append(np.linalg.norm(anchor - positive))
                negative_distance.append(np.linalg.norm(anchor - negative))
            except Exception as e:
                print(e)
                pdb.set_trace()
        print('[INFO] Mean distance of anchors with positive pairs is '
              '{} Max {} Min {}.\n[INFO] Mean distance of anchor with '
              'negative pairs is {} Max {} Min {}.'.format(
                    np.mean(positive_distance),
                    np.max(positive_distance),
                    np.min(positive_distance),
                    np.mean(negative_distance),
                    np.max(negative_distance),
                    np.min(negative_distance)))
