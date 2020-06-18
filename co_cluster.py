from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import AffinityPropagation


def type_consistent_cocluster(topic_word_dict0, ename2embed_bert, n_cluster_min, print_cls = False, save_file=None):
    topic_word_dict = {}
    all_words = []

    for topic in topic_word_dict0:
        topic_word_dict[topic] = []
        for ename in topic_word_dict0[topic]:
            if ename in ename2embed_bert:
                topic_word_dict[topic].append(ename)
                all_words.append(ename)
                
    topics = list(topic_word_dict0.keys())
#     print("topics")
#     print(topics)

    all_children = [x for x in all_words]
#     all_words.extend([x for x in topics if x in ename2embed_bert])
    all_embed = [ename2embed_bert[x][0] for x in all_words]
#     print(all_children)


    all_words_and_their_parents = []
    for word in all_words:
        for topic in topic_word_dict:
            if word in topic_word_dict[topic]:
                word0 = (topic, word)
                break
        all_words_and_their_parents.append(word0)
#     print(all_words_and_their_parents)
        

    # AP
    clustering = AffinityPropagation().fit(all_embed)
    n_clusters = max(clustering.labels_) + 1
    clusters = {}
    col_vectors = np.zeros( (len(topic_word_dict) ,n_clusters), dtype=float)
    for i in range(n_clusters):
        clusters[i] = [ all_words_and_their_parents[x] for x in range(len(clustering.labels_)) if clustering.labels_[x]==i]
        for word0 in clusters[i]:
            word0_col = int(word0[0])
            col_vectors[word0_col,i] = 1
    col_vectors = np.array(col_vectors)
    col_vectors += 0.1*np.ones( (len(topic_word_dict) ,n_clusters), dtype=int)

    
    
    for n_cluster in range(n_cluster_min, n_cluster_min+10):
    
        model = SpectralCoclustering(n_clusters=n_cluster, random_state=0)
        model.fit(col_vectors)

        new_topic_word_dict = {}
        coverage_list = []
        for ind in range(n_cluster):
            # print(ind)
            small_matrix = col_vectors[[x for x in range(len(model.row_labels_)) if model.row_labels_[x] == ind]]
            small_matrix = small_matrix[:,[x for x in range(len(model.column_labels_)) if model.column_labels_[x] == ind]]
            coverage_list.append(np.sum(small_matrix)/np.sum(np.ones_like(small_matrix)))
        if max(coverage_list) >= 0.7:
            break
            
    fit_data = col_vectors[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]
    
    cluster_count = [sum(model.row_labels_==x) for x in range(n_cluster)]
    # print("row cluster count: ", cluster_count)
    
    cluster_count = [sum(model.column_labels_==x) for x in range(n_cluster)]
    # print("column cluster count: ", cluster_count)

    
    coverage_thre = min(max(coverage_list), 0.4)
    # print('coverage: ',coverage_list)
    
    for ind in range(n_cluster):
        if coverage_list[ind] < coverage_thre:
            # print("del cluster ",ind)
            continue
        for topic in topic_word_dict:
            if model.row_labels_[int(topic)] == ind:
                new_topic_word_dict[topic] = [x for x in topic_word_dict[topic]]

    return new_topic_word_dict



def type_consistent(topic_word_dict0, ename2embed_bert, print_cls = False):
    topic_word_dict = {}
    all_words = []

    for topic in topic_word_dict0:
        topic_word_dict[topic] = []
        for ename in topic_word_dict0[topic]:
#             ename = ename.replace('_',' ')
            if ename in ename2embed_bert:
                topic_word_dict[topic].append(ename)
                all_words.append(ename)
                
    topics = list(topic_word_dict0.keys())

    all_words.extend([x for x in topics if x in ename2embed_bert])

    # all_words.extend(['POTENTIAL_PARENT_'+x for x in potential_parents if x in ename2embed_bert])
    # all_embed.extend([ename2embed_bert[x][0] for x in potential_parents if x in ename2embed_bert])

    all_children = []
    all_embed = []
    all_words_and_their_parents = []   
    for word in all_words:
        topic_count = 0
        word0 = word
        for topic in topic_word_dict:
            if word in topic_word_dict[topic]:
                word0 = '('+topic+')'+word0
                topic_count += 1
        if topic_count > 1:
            continue
        all_words_and_their_parents.append(word0)
        all_children.append(word)
        all_embed.append(ename2embed_bert[word][0])

    # AP
    clustering = AffinityPropagation().fit(all_embed)
    n_clusters = max(clustering.labels_) + 1
    clusters = {}
    singular_words = []
    for i in range(n_clusters):
        clusters[i] = [ all_words_and_their_parents[x] for x in range(len(clustering.labels_)) if clustering.labels_[x]==i]
        # if print_cls:
        #     print(clusters[i])
        category_count = set()
        for word0 in clusters[i]:
            tmp = word0.split('(')
            for seg in tmp:
                tmp2 = seg.split(')')
                if len(tmp2) < 2:
                    continue
                category_count.add(tmp2[0])
            if len(category_count) > 1:
                break
#         print(len(category_count))
        if len(category_count) <= 1:
            singular_words.extend([all_words[x] for x in range(len(clustering.labels_)) if clustering.labels_[x]==i])

#     print(singular_words)

    new_topic_word_dict = {}
    for topic in topic_word_dict:
        new_topic_word_dict[topic] = []
        for ename in topic_word_dict[topic]:
            if ename not in singular_words:
#                 ename = ename.replace(' ','_')
                new_topic_word_dict[topic].append(ename)

    return new_topic_word_dict

def type_consistent_for_list(l, topic_word_dict0, ename2embed_bert, print_cls = False):
    if len(l) < 60:
        return l
    tmp_cluster = {}
    tmp_cluster['0'] = l
    for topic in topic_word_dict0:
        tmp_cluster[topic] = topic_word_dict0[topic]
    tmp_cluster = type_consistent(tmp_cluster, ename2embed_bert, print_cls)    
    print(tmp_cluster['0'])
    l = tmp_cluster['0']
    return l

def get_cap_from_topics(topic_entities, word_cap, cap_list):
    if len(topic_entities) < 10:
        return topic_entities
    tmp = [x for x in topic_entities if word_cap[x] < max(cap_list) and word_cap[x] > min(cap_list)]
    start = 1.0
    end = 1.0
    while len(tmp) < 10:
        start += 0.1
        end -= 0.1
        tmp = [x for x in topic_entities if word_cap[x] < max(cap_list)*start and word_cap[x] > min(cap_list)*end]
        # print(tmp)
    return tmp
    
    
