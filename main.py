import time
import argparse
import numpy as np
import os
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from pytorch_transformers import *
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralCoclustering
from tqdm import tqdm
from random import sample
from utils import *
from model import *
from transfer import *
from co_cluster import *
from train import *
from batch_generation import *
from pytorch_transformers import *
from pytorch_transformers.modeling_bert import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BATCH_SIZE=16
TEST_BATCH_SIZE=512
EPOCHS = 5
max_seq_length = 128

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='dblp')
    parser.add_argument('--topic_file', default='topics_field.txt')
    parser.add_argument('--out_file', default='keyword_taxonomy.txt')


    args = parser.parse_args()
    print(args)
    dataset = args.dataset
    topic_file = args.topic_file



    # ent_sent_index.txt: record the sentence id where each entity occurs; used for generating BERT training sample
    print('------------------loading corpus!------------------')
    ent_sent_index = dict()
    with open(dataset+'/ent_sent_index.txt') as f:
        for line in f:
            ent = line.split('\t')[0]
            tmp = line.strip().split('\t')[1].split(' ')
            tmp = [int(x) for x in tmp]
            ent_sent_index[ent] = set(tmp)

    # sentences_.txt: sentence id to text
    sentences = dict()
    with open(dataset+'/sentences_.txt') as f:
        for i,line in enumerate(f):
            sentences[i] = line
            
    ent_ent_index = dict()
    with open(dataset+'/ent_ent_index.txt') as f:
        for line in f:
            ent = line.split('\t')[0]
            tmp = line.strip().split('\t')[1].split(' ')
            ent_ent_index[ent] = set(tmp)
            
    
    
    print('------------------loading embedding!------------------')

    pretrain = 0
    use_cap0 = False
    file = topic_file.split('_')[1].split('.')[0]

    # load word embedding
    word_emb, vocabulary, vocabulary_inv, emb_mat = get_emb(vec_file=os.path.join(dataset, 'emb_part_'+file + '_w.txt'))

    # load topic embedding
    topic_emb, topic2id, id2topic, topic_hier = get_temb(vec_file=os.path.join(dataset, 'emb_part_'+file+'_t.txt'), topic_file=os.path.join(dataset, topic_file))

    # load word specificity
    word_cap = get_cap(vec_file=os.path.join(dataset, 'emb_part_'+file+'_cap.txt'))

    ename2embed_bert = loadEnameEmbedding(os.path.join(dataset, 'BERTembed.txt'), 768)

    print('------------------generating subtopic candidates!------------------')
    # calculate topic representative words: rep_words
    rep_words = {}
    for topic in topic_emb:
        print(topic)
        sim_ranking = topic_sim(topic, vocabulary_inv, topic_emb, word_emb)
        if pretrain:
            cap_ranking = np.ones((len(vocabulary)))
            word_cap1 = np.ones((len(vocabulary)))
        else:
            cap_ranking = rank_cap(word_cap, vocabulary_inv, topic)
        if use_cap0:
            rep_words[topic] = aggregate_ranking(sim_ranking, cap_ranking, word_cap, topic, vocabulary_inv, pretrain, ent_sent_index, word_cap[topic])
        else:
            rep_words[topic] = aggregate_ranking(sim_ranking, cap_ranking, word_cap, topic, vocabulary_inv, pretrain, ent_sent_index)
    rep_words1 = {}
    for topic in topic_emb:
        rep_words1[topic] = [x for x in rep_words[topic]]
    for word in rep_words:
        rep_words[word] = [word]


    print('------------------initializing relation classifier!------------------')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RelationClassifer.from_pretrained('bert-base-uncased')
    model.float()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)

    print('------------------generating training data!------------------')

    # generating training and testing data
    total_data, sentences_index = process_training_data(sentences, rep_words, topic_hier, max_seq_length, ent_sent_index, tokenizer)
    train_data = total_data[:int(len(total_data)/2*0.95)]
    train_data.extend(total_data[int(len(total_data)/2):int(len(total_data)/2+len(total_data)/2*0.95)])
    valid_data = total_data[int(len(total_data)/2*0.95):int(len(total_data)/2)]
    valid_data.extend(total_data[int(len(total_data)/2*0.95+len(total_data)/2):])
    # test_data = process_test_data(rep_words[test_topic], test_cand, max_seq_length)
    print(f"training data point number: {len(train_data)}")

    # training the bert classifier
    print('------------------training relation classifier!------------------')

    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_func(train_data, model, BATCH_SIZE, optimizer, scheduler, generate_batch)
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        
        valid_loss, valid_acc = valid_func(valid_data, model, BATCH_SIZE, generate_batch)
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
        
        
        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))    


    print('------------------extracting subtopic candidates!------------------')
    
    entity_ratio_alltopics = {}
    entity_count_alltopics = {}

    training_topic = {}
    for topic in topic_hier['ROOT']:
        if topic in topic_hier:
            training_topic[topic] = word_cap[topic]
    training_topic = sorted(training_topic.items(), key = lambda x: x[1])
    train_topic = training_topic[0][0]

    for test_topic in topic_hier['ROOT']:
        
        sim_ranking = topic_sim(test_topic, vocabulary_inv, topic_emb, word_emb)
        cap_ranking, target_cap = rank_cap_customed(word_cap, vocabulary_inv, [vocabulary[word] for word in topic_hier[train_topic]])
        coefficient = max(word_cap[test_topic] / word_cap[train_topic],1)
        test_cand = aggregate_ranking(sim_ranking, cap_ranking, word_cap, test_topic, vocabulary_inv, pretrain, ent_sent_index, target_cap*coefficient)

        test_data = process_test_data(sentences, rep_words[test_topic], test_cand, max_seq_length,ent_sent_index,ename2embed_bert,  tokenizer)
        print(f"test data point number: {len(test_data)}")
        if len(test_data) > 10000:
            test_data = sample(test_data, 10000)

        entity_ratio, entity_count = relation_inference(test_data, model, TEST_BATCH_SIZE)
        entity_ratio_alltopics[test_topic] = entity_ratio
        entity_count_alltopics[test_topic] = entity_count

    child_entities_count = sum_all_rel(topic_hier['ROOT'], entity_count_alltopics, mode='child')

    # print(child_entities_count)

    child_entities = type_consistent(child_entities_count, ename2embed_bert)

    print('------------------Topic-Type Matrix Creation!------------------')
    

    clusters_all = {}
    k=0
    start_list = [0]
    for j,topic in enumerate(topic_hier['ROOT']): 
        
        X = []
        for ent in child_entities[topic]:
            if ent not in word_emb:
                continue
            X.append(word_emb[ent])
        X = np.array(X)

        clustering = AffinityPropagation().fit(X)
        n_clusters = max(clustering.labels_) + 1
        clusters = {}
        for i in range(n_clusters):
            clusters[str(i)] = [child_entities[topic][x] for x in range(len(clustering.labels_)) if clustering.labels_[x] == i]
            
            clusters_all[str(k)] = clusters[str(i)]
            k+=1
        start_list.append(k)

    new_clusters = type_consistent_cocluster(clusters_all, ename2embed_bert, n_cluster_min = 8, print_cls = True)

    tmp = defaultdict(list)

    print('------------------Subtopics found!------------------')

    topic_idx = 0
    for k in range(len(clusters_all)):
        if k >= start_list[topic_idx]:
            print('\n',topic_hier['ROOT'][topic_idx])
            topic_idx += 1
        if str(k) in new_clusters and len(new_clusters[str(k)]) > 1:
            print(new_clusters[str(k)])
            tmp[topic_hier['ROOT'][topic_idx-1]].append(new_clusters[str(k)])

    child_entities = tmp

    print('------------------Root Node Candidate Generation!------------------')
    parent_cand = get_common_ent_for_list(topic_hier['ROOT'],ent_ent_index)
    if len(parent_cand) > 1000:
        parent_cand = type_consistent_for_list(parent_cand, rep_words, ename2embed_bert, False)

    parent_entity_ratio_alltopics = {}
    parent_entity_count_alltopics = {}
    for test_topic in topic_hier['ROOT']:
        print(f'test topic: {test_topic}')
        
        test_data = process_test_data(sentences, [test_topic], list(parent_cand), max_seq_length,ent_sent_index, ename2embed_bert, tokenizer)
        print(f"test data point number: {len(test_data)}")
        
        # if len(test_data) > 10000:
        #     test_data = sample(test_data, 10000)       
        entity_ratio, entity_count = relation_inference(test_data, model, TEST_BATCH_SIZE,mode='child')
        parent_entity_ratio_alltopics[test_topic] = entity_ratio
        parent_entity_count_alltopics[test_topic] = entity_count

    parent_entities_count = sum_all_rel(topic_hier['ROOT'], parent_entity_count_alltopics, mode='parent')
    parent_result = get_threshold_from_dict(parent_entities_count, 1/2)
    parent_result = type_consistent_for_list(parent_result, rep_words, ename2embed_bert, False)
    print(f'Discover {len(parent_result)} root nodes!')
    print(parent_result)
    

    print('------------------New topic finding!------------------')
    topic_cand = defaultdict(int)
    for topic in parent_result:
        for ent in ent_ent_index[topic]:
            topic_cand[ent] += 1
    topic_cand = [x for x in topic_cand if topic_cand[x] >= len(parent_result)/2]
    
    remove_list = []
    for topic in child_entities_count:
        remove_list.extend(child_entities_count[topic])
    remove_list.extend(parent_result)

    tmp = []
    for topic in topic_cand:
        if topic not in remove_list:
            tmp.append(topic)
    topic_cand = tmp
    
    topic_entity_ratio_alltopics = {}
    topic_entity_count_alltopics = {}
    for test_topic in parent_result:
        print(f'test topic: {test_topic}')        
        test_data = process_test_data(sentences, [test_topic], list(topic_cand), max_seq_length,ent_sent_index, ename2embed_bert, tokenizer)
        print(f"test data point number: {len(test_data)}")
        if len(test_data) > 10000:
            test_data = sample(test_data, 10000)    
        
        entity_ratio, entity_count = relation_inference(test_data, model, TEST_BATCH_SIZE,mode='child')
        topic_entity_ratio_alltopics[test_topic] = entity_ratio
        topic_entity_count_alltopics[test_topic] = entity_count

    topic_entities_count = sum_all_rel(parent_result, topic_entity_count_alltopics, mode='child')


    topic_entities = get_threshold_from_dict(topic_entities_count, 1/3)
    cap_list = [word_cap[x] for x in topic_hier['ROOT']]
    print([(x, word_cap[x]) for x in topic_entities if x in word_cap])
    topic_entities = get_cap_from_topics(topic_entities, word_cap, cap_list)
    for t in topic_hier['ROOT']:
        if t in topic_hier:
            for t1 in topic_hier[t]:
                if t1 in topic_entities:
                    topic_entities.remove(t1)

    # topic_entities = [x for x in topic_entities if word_cap[x] < max(cap_list) and word_cap[x] > min(cap_list)]
    # topic_entities = type_consistent_for_list(topic_entities, rep_words, ename2embed_bert, False)
    # print(topic_entities)
    for t in topic_hier['ROOT']:
        if t in topic_hier:
            for t1 in topic_hier[t]:
                if t1 in topic_entities:
                    topic_entities.remove(t1)
        for t1 in child_entities[t]:
            if t1 in topic_entities:
                topic_entities.remove(t1)
    print(topic_entities)


    print('------------------Subtopic finding for new topics!------------------')
    
    topic_hier1 = {}

    topic_hier1['ROOT']= topic_entities
    for topic in topic_hier:
        if topic == 'ROOT':
            for t in topic_hier[topic]:
                if t not in topic_hier1[topic]:
                    topic_hier1[topic].append(t)
        else:
            topic_hier1[topic] = [x for x in topic_hier[topic]]
    # print(topic_hier)
    save_tree_to_file(topic_hier1, 'intermediate.txt')

    entity_ratio_alltopics1 = {}
    entity_count_alltopics1 = {}

    for test_topic in topic_hier1['ROOT']:
        if test_topic in topic_hier['ROOT']:
            entity_ratio_alltopics1[test_topic] = entity_ratio_alltopics[test_topic]
            entity_count_alltopics1[test_topic] = entity_count_alltopics[test_topic]
            continue
        
        sim_ranking = topic_sim(test_topic, vocabulary_inv, topic_emb, word_emb)
        cap_ranking, target_cap = rank_cap_customed(word_cap, vocabulary_inv, [vocabulary[word] for word in topic_hier[train_topic]])
        coefficient = max(word_cap[test_topic] / word_cap[train_topic],1)
        test_cand = aggregate_ranking(sim_ranking, cap_ranking, word_cap, test_topic, vocabulary_inv, pretrain, ent_sent_index, target_cap*coefficient)
        print(f'test topic: {test_topic}')    
        test_data = process_test_data(sentences, [test_topic], test_cand, max_seq_length,ent_sent_index, ename2embed_bert, tokenizer)
        print(f"test data point number: {len(test_data)}")
        
        entity_ratio, entity_count = relation_inference(test_data, model, TEST_BATCH_SIZE)
        entity_ratio_alltopics1[test_topic] = entity_ratio
        entity_count_alltopics1[test_topic] = entity_count

    child_entities_count1 = sum_all_rel(topic_hier1['ROOT'], entity_count_alltopics1, mode='child')

    child_entities1 = type_consistent(child_entities_count1, ename2embed_bert)

    for ent in topic_hier1['ROOT']:
        if ent not in child_entities1:
            topic_hier1['ROOT'].remove(ent)

    clusters_all = {}
    k=0
    start_list = [0]
    for j,topic in enumerate(topic_hier1['ROOT']):   
        X = []
        for ent in child_entities1[topic]:
            if ent not in word_emb:
                continue
            X.append(word_emb[ent])
        if len(X) == 0:
            continue
        X = np.array(X)

        clustering = AffinityPropagation().fit(X)
        n_clusters = max(clustering.labels_) + 1
        clusters = {}
        for i in range(n_clusters):
            clusters[str(i)] = [child_entities1[topic][x] for x in range(len(clustering.labels_)) if clustering.labels_[x] == i]
            
            clusters_all[str(k)] = clusters[str(i)]
            k+=1
        start_list.append(k)
    new_clusters = type_consistent_cocluster(clusters_all, ename2embed_bert, n_cluster_min = 2, print_cls = True, save_file='dblp_field+_cls8')

    print(start_list)

    tmp = defaultdict(list)

    topic_idx = 0
    for k in range(len(clusters_all)):
        if k >= start_list[topic_idx]:
    #         print('\n',topic_hier1['ROOT'][topic_idx])
            topic_idx += 1
        if str(k) in new_clusters and len(new_clusters[str(k)]) > 1:
    #         print(new_clusters[str(k)])
            tmp[topic_hier1['ROOT'][topic_idx-1]].append(new_clusters[str(k)])

    child_entities1 = tmp
    for t in topic_hier['ROOT']:
        child_entities1[t] = child_entities[t]

    print('------------------Outputing the topical taxonomy!------------------')
        
    for t in topic_hier1['ROOT']:
        if len(child_entities1[t]) == 0:
            continue
        print(t)
        for l in child_entities1[t]:
            print(l)
        print('')

    # print the keyword taxonomy, nodes in which will be enriched later by concept learning.
    with open(os.path.join(dataset, args.out_file), 'w') as fout:
        for topic in topic_hier1['ROOT']:  
            if len(child_entities1[topic]) > 0:      
                fout.write(topic+'\n')
                for cls in child_entities1[topic]:
                    fout.write(' '.join(cls)+'\n')
                fout.write('\n')

    for topic in topic_hier1['ROOT']:
        if len(child_entities1[topic]) > 0:
            with open(os.path.join(dataset, 'topics_'+topic+'.txt'),'w') as fout:
                for cls in child_entities1[topic]:
                    fout.write(' '.join(cls)+'\n')

            

