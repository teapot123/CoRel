from collections import defaultdict
import numpy as np
import os
import torch
from tqdm import tqdm

def load_seed(dataset, file):
    topic_words = {}
    with open(dataset+'/result_'+file+'.txt') as f:
        data=f.readlines()
        current_topic = ''
        for line in data:
            if len(line.strip()) == 0:
                current_topic = ''
                continue
            elif len(line.split(' ')) == 1:
                current_topic = line.split(':')[0]
                continue
            elif current_topic != '':
                topic_words[current_topic] = line.strip().split(' ')

    return topic_words

def get_emb(vec_file):
    f = open(vec_file, 'r')
    contents = f.readlines()[1:]
    word_emb = {}
    vocabulary = {}
    vocabulary_inv = {}
    emb_mat = []
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        word = tokens[0]
        vec = tokens[1:]
        vec = [float(ele) for ele in vec]
        word_emb[word] = np.array(vec)
        vocabulary[word] = i
        vocabulary_inv[i] = word
        emb_mat.append(np.array(vec))
    vocab_size = len(vocabulary)
    emb_mat = np.array(emb_mat) 
    return word_emb, vocabulary, vocabulary_inv, emb_mat

def get_temb(vec_file, topic_file):
    topic2id = {}
    topic_emb = {}
    id2topic = {}
    topic_hier = {}
    i = 0
    with open(topic_file, 'r') as f:
        for line in f:
            parent = line.strip().split('\t')[0]
            temp = line.strip().split('\t')[1]          
            for topic in temp.split(' '):
                topic2id[topic] = i
                id2topic[i] = topic
                i += 1
                if parent not in topic_hier:
                    topic_hier[parent] = []
                topic_hier[parent].append(topic)
    f = open(vec_file, 'r')
    contents = f.readlines()[1:]
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        vec = tokens
        vec = [float(ele) for ele in vec]
        topic_emb[id2topic[i]] = np.array(vec)
    return topic_emb, topic2id, id2topic, topic_hier

def get_cap(vec_file, cap0_file=None):
    f = open(vec_file, 'r')
    contents = f.readlines()[1:]
    word_cap = {}
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        word = tokens[0]
        vec = tokens[1]
        vec = float(vec)
        word_cap[word] = vec
    
    if cap0_file is not None:
        with open(cap0_file) as f:
            contents = f.readlines()[1:]
            for i, content in enumerate(contents):
                content = content.strip()
                tokens = content.split(' ')
                word = tokens[0]
                vec = tokens[1]
                vec = float(vec)
                word_cap[word] = vec 
    return word_cap

def topic_sim(query, idx2word, t_emb, w_emb):
    if query in t_emb:
        q_vec = t_emb[query]
    else:
        q_vec = w_emb[query]
    word_emb = np.zeros((len(idx2word), 100))
    for i in range(len(idx2word)):
        word_emb[i] = w_emb[idx2word[i]]
    res = np.dot(word_emb, q_vec)
    res = res/np.linalg.norm(word_emb, axis=1)
    sort_id = np.argsort(-res)

    return sort_id

def rank_cap(cap, idx2word, class_name):
    word_cap = np.zeros(len(idx2word))
    for i in range(len(idx2word)):
        if idx2word[i] in cap:
            word_cap[i] = (cap[idx2word[i]]-cap[class_name]) ** 2
        else:
            word_cap[i] = np.array([1.0])
    low2high = np.argsort(word_cap)
    return low2high

def rank_cap_customed(cap, idx2word, class_idxs):
    target_cap = np.mean([cap[idx2word[ind]] for ind in class_idxs])
    word_cap = np.zeros(len(idx2word))
    for i in range(len(idx2word)):
        if idx2word[i] in cap:
            word_cap[i] = (cap[idx2word[i]]-target_cap) ** 2
        else:
            word_cap[i] = np.array([1.0])
    low2high = np.argsort(word_cap)

    return low2high, target_cap

def aggregate_ranking(sim, cap, word_cap, topic, idx2word, pretrain, ent_sent_index, target=None):
    simrank2id = np.ones(len(sim)) * np.inf
    caprank2id = np.ones(len(sim)) * np.inf
    for i, w in enumerate(sim[:]):
        simrank2id[w] = i + 1
    for i, w in enumerate(cap):
        if pretrain == 0:
            if target is not None and word_cap[idx2word[w]] > target:
                caprank2id[w] = i + 1
            if target is None:
                caprank2id[w] = i + 1
    if pretrain == 0:        
        agg_rank = simrank2id * caprank2id
        final_rank = np.argsort(agg_rank)
        final_rank_words = [idx2word[idx] for idx in final_rank[:500] if idx2word[idx] in ent_sent_index]
    else:
        agg_rank = simrank2id
        final_rank = np.argsort(agg_rank)
        final_rank_words = [idx2word[idx] for idx in final_rank[:500] if idx2word[idx] in ent_sent_index]
    # print(final_rank_words)  
    return final_rank_words


def get_common_ent_for_list(l, ent_ent_index):

    parent_cand = set()

    for test_topic in l:
        if len(parent_cand) == 0:
            parent_cand = ent_ent_index[test_topic]
        else:
            parent_cand = parent_cand.intersection(ent_ent_index[test_topic])

    return parent_cand

def get_common_ent_for_list_with_dict(l,d):
    parent_result = set()
    for test_topic in l:
        if len(parent_result) == 0:
            parent_result = set(d[test_topic])
        else:
            parent_result = parent_result.intersection(set(d[test_topic]))

    return parent_result

def get_threshold_from_dict(d, thre):
    parent_result_entities = defaultdict(int)
    for topic in d:
        for ent in d[topic]:
            parent_result_entities[ent] += 1
    # print(parent_result_entities)
    parent_result_entities = [x for x in parent_result_entities if parent_result_entities[x] >= len(d)*thre]
    # print(parent_result_entities)
    return parent_result_entities

def loadEnameEmbedding(filename, dim=100, header=False):
    """ Load the entity embedding with word as context

    :param filename:
    :param dim: embedding dimension
    :return:
    """

    M = dim  # dimensionality of embedding
    ename2embed = {}
    with open(filename, "r") as fin:
        if header:
            next(fin)
        for line in fin:
            seg = line.strip().split()
            word = seg[0:-M]
            del seg[1:-M]
            seg[0] = '_'.join( word )
            embed = np.array([float(ele) for ele in seg[1:]])
            ename2embed[seg[0]] = embed.reshape(1, M)


    return ename2embed

def save_tree_to_file(topic_hier, fileName):
    with open(fileName, 'w') as fout:
        for k in topic_hier:
            fout.write(k)
            for v in topic_hier[k]:
                fout.write('\t'+v)
            fout.write('\n')

            
