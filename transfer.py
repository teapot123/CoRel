from train import *
from batch_generation import *
import numpy as np
from collections import defaultdict

def kl_divergence(p,q):
    d = 0
    num_labels = len(p)
    for i in range(num_labels):
        d += p[i] * np.log(p[i]/q[i])
    return d

def relation_inference(test_data, model, TEST_BATCH_SIZE,mode='child'):
    logits, cand_entities = test(test_data, model, TEST_BATCH_SIZE, generate_test_batch)
    
    if(len(test_data)==0):
        return {},{}

    logits = np.array(logits)
    labels = np.argmax(logits, axis=1)
    test_num = len(labels)
    entity_ratio = {}
    entity_count = {}

    for i in range(int(test_num/2)):
        ent = cand_entities[2*i]
        l1 = labels[2*i]
        kl1 = kl_divergence([1/3,1/3,1/3],logits[2*i])
        l2 = labels[2*i + 1]
        kl2 = kl_divergence([1/3,1/3,1/3],logits[2*i+1])
        if ent not in entity_ratio:
            entity_ratio[ent] = np.zeros((3))
        entity_ratio[ent] += logits[2*i]
        entity_ratio[ent] += [logits[2*i+1][1], logits[2*i+1][0], logits[2*i+1][2]]
        if kl1 > 0.5 and kl2 > 0.5:
            if mode == 'child':
                if l1 == 0 and l2 == 1 and logits[2*i][0]>0.7 and logits[2*i+1][1]>0.7:
        #             print(f'parent: {cand_entities[2*i]}')
                    if ent not in entity_count:
                        entity_count[ent] = np.zeros((3))
                    entity_count[ent][0] += 1
                elif l1 == 1 and l2 == 0 and logits[2*i][1]>0.7 and logits[2*i+1][0]>0.7:
                    if ent not in entity_count:
                        entity_count[ent] = np.zeros((3))
                    entity_count[ent][1] += 1
    #                 print(f'child: {cand_entities[2*i]} {logits[2*i]} {kl1} {kl2}')
                elif l1 == 2 and l2 == 2:
                    if ent not in entity_count:
                        entity_count[ent] = np.zeros((3))
                    entity_count[ent][2] += 1
            elif mode == 'parent':
                if l1 == 0 and l2 == 1 :
        #             print(f'parent: {cand_entities[2*i]}')
                    if ent not in entity_count:
                        entity_count[ent] = np.zeros((3))
                    entity_count[ent][0] += 1
                elif l1 == 1 and l2 == 0 :
                    if ent not in entity_count:
                        entity_count[ent] = np.zeros((3))
                    entity_count[ent][1] += 1
    #                 print(f'child: {cand_entities[2*i]} {logits[2*i]} {kl1} {kl2}')
                elif l1 == 2 and l2 == 2:
                    if ent not in entity_count:
                        entity_count[ent] = np.zeros((3))
                    entity_count[ent][2] += 1
    #             print(f'no relation: {cand_entities[2*i]}')
    #         print(f"kl1: {kl1} kl2: {kl2}")
    return entity_ratio, entity_count

def sum_all_rel(test_topics, entity_count_alltopics, mode='child'):
    child_entities_count = defaultdict(list)
    entity_confidence = defaultdict(float)
    for test_topic in test_topics:
#         print(f"test topic: {test_topic}")
        entity_count = entity_count_alltopics[test_topic]

        for ent in entity_count:            
            ratio = entity_count[ent]/np.sum(entity_count[ent])
            # print(f"{ent}: parent: {ratio[0]} child: {ratio[1]} no relation: {ratio[2]}")
            if mode == 'child' and ratio[1] > 0.7:
                child_entities_count[test_topic].append(ent)
                entity_confidence[ent] += ratio[1]
            elif mode == 'parent' and ratio[0] > 0.7:
                child_entities_count[test_topic].append(ent)
                entity_confidence[ent] += ratio[0]
    for ent in entity_confidence:
        entity_confidence[ent] /= len(test_topics)
    entity_conf = sorted(entity_confidence.items(), key = lambda x: x[1], reverse = True)
    # print(entity_conf)
    return child_entities_count
