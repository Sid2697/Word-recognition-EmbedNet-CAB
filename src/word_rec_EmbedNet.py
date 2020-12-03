"""
This file has the code for calculating word accuracy using word embedding information
"""

# Standard library imports
import pdb
import time
import pickle
import argparse

# Third party imports
import tqdm
import torch
import numpy as np
import torch.nn as nn
import Levenshtein as lev
from sklearn.preprocessing import normalize
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from models import EmbedNet

parser = argparse.ArgumentParser()

# Embeddings and text file paths
parser.add_argument('--image_embeds', default='embeddings/topk_preds_100featsImg.npy', help='path to the image embeddings')
parser.add_argument('--topk_embeds', default='embeddings/topk_preds_100featsSynth.npy', help='path to the topk text embeds')
parser.add_argument('--predictions_file', default='gen_files/top_preds_embeds_100_with_conf.txt', help='path to the top preds text file options: [top_preds_embeds_with_confidence_1500, top_preds_embeds_all_with_confidence, top_preds_embeds_all_with_confidence_telugu_deep]')
parser.add_argument('--image_file', default='gen_files/image_embed_top_k_100.txt', help='path to the text file used for producing image embeddings options: [image_embed_top_k_1500, image_embed_top_k_all, test_ann_1000_pages_Telugu_deep]')

# Different experiments' flags
parser.add_argument('--use_confidence', default=False, action="store_true", help='If True we will use confidence score for re-ranking')
parser.add_argument('--cab', default=False, action="store_true", help='If True we will use CAB for improving the word recognition accuracy')
parser.add_argument('--cab_alpha', default=33, type=int, help='Hyperparameter alpha defined for the CAB module')
parser.add_argument('--cab_beta', default=1, type=int, help='Hyperparameter beta defined for the CAB module')

# Flags related to NN experiments
parser.add_argument('--use_model', default=False, action="store_true", help='Wheater to use or not use NN')
parser.add_argument('--in_features', default=2048, type=int, help='Size of the input to the neural network')
parser.add_argument('--out_features', default=128, type=int, help='Size of the output of the neural network')
parser.add_argument('--hidden_layers', nargs='+', type=int, default=[1024, 512, 256, 128], help='List of input size of the hidden layers')
parser.add_argument('--model_path', default='models/WNet1AdamLR000001EXTOnGen1MarginNoConfidence240620_accuracy.pkl', help='Path to the model to use')
parser.add_argument('--test_split', default=0.8, type=float, help='Split for testing the trained model on un-seen data')
parser.add_argument("--testing", default=False, action="store_true" , help="Activate testing mode")

parser.add_argument('--k', default=20, type=int, help='Value of K')
parser.set_defaults(features=False)
args = parser.parse_args()
print(args)

with open(args.predictions_file) as file:
    fileData = file.readlines()

if not args.use_confidence:
    assert not args.cab, "CAB only works if use_confidence is True"

predictions = [item.split()[-3] for item in fileData]
if args.use_confidence:
    confidenceScores = [float(item.split()[-2]) for item in fileData]

with open(args.image_file) as file:
    file_data = file.readlines()
query = [item.split()[-3] for item in file_data]

print("[INFO] Loading word image and predictions' embeddings...")
image_embeds_orig = np.load(args.image_embeds, mmap_mode='r')    # Enabling mmap_mode uses very very less RAM for loading the array as it uses the array directly from the disk
try:
    topk_embeds_orig = np.load(args.topk_embeds, mmap_mode='r')
except OSError:
    # This is for using the memory mapped array which can be used to handle large arrays in numpy.
    topk_embeds_orig = np.memmap(args.topk_embeds, dtype=np.float32, mode='r', shape=(2109500, 2048))


def get_EmbedNet_embed(input_embedding):
    print('[INFO] Using EmbedNet...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EmbedNet(args.in_features, args.out_features, args.hidden_layers).to(device)
    if device.type == 'cpu':
        checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval().double()
    loader = DataLoader(dataset=np.array(input_embedding), num_workers=1, batch_size=1024, shuffle=False, pin_memory=True)
    final_embedding = np.zeros((input_embedding.shape[0], args.out_features))
    ko = 0
    k = 1024
    for embedding in tqdm.tqdm(loader, desc='[INFO] NN pass'): # Generating one embedding at a time
        if device.type == 'cpu':
            output = model(embedding.double())
        else:            
            output = model(embedding.cuda().double())
        final_embedding[ko:k, :] = output.cpu().detach().numpy()
        ko += 1024
        k += 1024
    return final_embedding


if args.testing:
    print('[INFO] Evaluating only on the test set...')
    image_test_count = int(image_embeds_orig.shape[0] - np.round(image_embeds_orig.shape[0] * args.test_split))
    text_test_count  = image_test_count * 20
    image_embeds = image_embeds_orig[-image_test_count:]
    topk_embeds = topk_embeds_orig[-text_test_count:]
    query = query[-image_test_count:]
    predictions = predictions[-text_test_count:]

if args.use_model:
    print('[INFO] Using model...')
    if args.testing:
        image_embeds = get_EmbedNet_embed(image_embeds)
        topk_embeds = get_EmbedNet_embed(topk_embeds)
    else:
        image_embeds = get_EmbedNet_embed(image_embeds_orig)
        topk_embeds = get_EmbedNet_embed(topk_embeds_orig)
else:
    image_embeds = image_embeds_orig
    topk_embeds = topk_embeds_orig

if args.cab and args.use_confidence:
    print('[INFO] Using the CAB module with alpha = {} and beta = {}...'.format(args.cab_alpha, args.cab_beta))

accuracyList = list()   # List for holding the accuracies
for i in range(args.k):     # Looping over top k predictions
    topk_count = 0  # Keeping track of TopK number
    correct = 0 # Keeping track of correct words
    total = 0   # Keeping track of total words tested
    use_ocr = 0
    use_other = 0
    # Looping over for calculating K for all K = 1, 2, ... K
    for count in tqdm.tqdm(range(len(image_embeds)), desc='[INFO] K = {}'.format(i + 1)):
        total += 1
        first_img_embed = image_embeds[count]   # Getting the first embedding
        corrs_topk_embeds = topk_embeds[topk_count : topk_count + i + 1]    # Getting top k embeddings corresponding to the first embedding
        kdt = KDTree(corrs_topk_embeds, leaf_size=30, metric='euclidean')   # Creating the KDTree for querying
        dist, ind = kdt.query(first_img_embed.reshape(1, -1), k=corrs_topk_embeds.shape[0], dualtree=True)  # Getting the distance and index by querying first embed on corresponding text
        # If we want to use the confidence scores
        if args.use_confidence:
            conf = list() # List for keeping track of the confidence scores
            for confCount in range(len(dist[0])):
                conf.append(confidenceScores[topk_count + ind[0][confCount]])
            if args.cab:    # If selected, using the CAB module
                conf_ = [(1 - item) * args.cab_alpha for item in conf]
                updatedDist = conf_ + dist[0] * args.cab_beta
            else:
                updatedDist = conf + dist[0] # Updated distace value after considering the confidence scores
            newInd = ind[0][np.where(min(updatedDist) == updatedDist)[0][0]] # Updated index value after considering the confidence scores                
            pred = predictions[topk_count + newInd]    # Updated predictions after considering the confidence scores
        # If we are not using the confidence scores
        else:
            pred = predictions[topk_count + ind[0][0]]
        gt = query[count] # Getting the ground truth
        # Checking if the predicion equals the ground truth
        if lev.distance(gt, pred) == 0:
            correct += 1
        # Updating the top k count
        topk_count += 20
    accuracyList.append(correct/total * 100)
accuracyList = [round(item, 3) for item in accuracyList]
print('[INFO] Top {} accuracies are: {}.'.format(len(accuracyList), accuracyList))
print('[INFO] Number of words tested on {}.'.format(total))

# Command using for generating final new results (02/12/20)
# python3 src/word_rec_EmbedNet.py --image_embeds embeddings/topk_preds_100featsImg.npy --topk_embeds embeddings/topk_preds_100featsSynth.npy --predictions_file gen_files/top_preds_embeds_100_with_conf.txt --image_file gen_files/image_embed_top_k_100.txt --use_model  --model_path /ssd_scratch/cvit/sid/WNet1AdamLR000001EXTOnGen1MarginNoConfidence240620.pkl --hidden_layers 1024 --test_split 1 --testing
# Command updated on 03/12/20
# python3 src/word_rec_EmbedNet.py  --use_model  --hidden_layers 1024
# Command for running baseline model
# python3 src/word_rec_EmbedNet.py
# Command for running model using the confidence scores
# python3 src/word_rec_EmbedNet.py --use_confidence
# Command for running model using the EmbedNet
# python3 src/word_rec_EmbedNet.py --use_confidence --use_model --hidden_layers 1024
# Command for running model using EmbedNet and CAB
# python3 src/word_rec_EmbedNet.py --use_confidence --use_model --hidden_layers 1024 --cab
