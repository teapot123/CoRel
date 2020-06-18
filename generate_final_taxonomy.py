import numpy as np
import os
import argparse




def get_emb(vec_file):
    print(vec_file)
    f = open(vec_file, 'r')
    contents = f.readlines()[1:]
    word_emb = {}
    vocabulary = {}
    vocabulary_inv = {}
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        word = tokens[0]
        vec = tokens[1:]
        vec = [float(ele) for ele in vec]
        word_emb[word] = np.array(vec)
        vocabulary[word] = i
        vocabulary_inv[i] = word
    return word_emb, vocabulary, vocabulary_inv


def get_cap(vec_file):
    print(vec_file)
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

    return word_cap


def build_emb_mat(word_emb):
    vocab = {}
    emb_mat = []
    i = 0
    for word in word_emb:
        vocab[i] = word
        emb_mat.append(word_emb[word])
        i += 1
    emb_mat = np.array(emb_mat)
    return vocab, emb_mat

def calc_sim(word_emb, word_pair):
    w1 = word_emb[word_pair[0]]
    w2 = word_emb[word_pair[1]]
    return np.dot(w1, w2)/np.linalg.norm(w1)/np.linalg.norm(w2)

def calc_sim2(topic_emb, word_emb, word_pair):
    # print(word_pair)
    w1 = topic_emb[word_pair[0]]
    w2 = word_emb[word_pair[1]]
    return np.dot(w1, w2)/np.linalg.norm(w1)/np.linalg.norm(w2)


def most_sim(query, id2topic, idx2word, t_emb, w_emb, print_num, cap, thre):
    q_vec = t_emb[query]
    word_emb = np.zeros((len(idx2word), 100))
    word_cap = np.zeros(len(idx2word))
    for i in range(len(idx2word)):
        word_emb[i] = w_emb[idx2word[i]]
        word_cap[i] = cap[idx2word[i]]
    res = np.dot(word_emb, q_vec)
    res = res/np.linalg.norm(word_emb, axis=1)
    sort_id = np.argsort(-res)
    # res2= [res[sort_id[i]] for i in range(print_num)]
    sim_sort = [sort_id[i] for i in range(print_num*2)]
    semantic_sort = np.argsort([word_cap[sort_id[i]] for i in range(print_num)])

    print(f'Most similar {print_num} words with topic {id2topic[query]}:')
    rank_seman = [(idx2word[sort_id[i]], word_cap[sort_id[i]]) for i in range(print_num)]
    rank_seman = [(idx2word[sim_sort[semantic_sort[i]]], word_cap[sim_sort[semantic_sort[i]]]) for i in range(print_num)]
    print([f"{ele[0]}: {ele[1]}" for ele in rank_seman])
    #print(res2)
    return rank_seman


def topic_sim(query, idx2word, t_emb, w_emb):
    q_vec = t_emb[query.replace(' ','')]
    word_emb = np.zeros((len(idx2word), 100))
    for i in range(len(idx2word)):
        word_emb[i] = w_emb[idx2word[i]]
    res = np.dot(word_emb, q_vec)
    res = res/np.linalg.norm(word_emb, axis=1)
    sort_id = np.argsort(-res)

    return sort_id


def seed_topic_distr(id2topic, topic_emb, word_emb):

    for i in range(len(id2topic)):
        word_topic_distr = []
        print(f"\nWord \"{id2topic[str(i)]}\" topic similarity:")
        for j in range(len(id2topic)):
            sim = calc_sim2(topic_emb, word_emb, (str(j), id2topic[str(i)]))
            word_topic_distr.append(sim)
        print(word_topic_distr)
    return


def topic_coherence(topic, id2topic, vocabulary_inv, topic_emb, word_emb, num, test, cap, thre):
    #print('function topic_coherence')
    #print(word_emb["8086"])
    word_list = most_sim(topic, id2topic, vocabulary_inv, topic_emb, word_emb, num, cap, thre)
    tmp=word_list[0:5]
    tmp.extend(word_list[10:15])
    word_group_plus = [word_list[0:10], word_list[5:], tmp]
    word_group_minus = [word_list[0:5], word_list[5:10], word_list[10:15]]
    if test == 'local':
        get_str="http://palmetto.aksw.org/palmetto-webapp/service/cv?words="
    else:
        get_str="http://palmetto.aksw.org/palmetto-webapp/service/umass?words="
    coher = 0
    '''for group in word_group_plus:
        url = get_str + '%20'.join(group)
        coher+=requests.get(url=url).json()
    for group in word_group_minus:
        url = get_str + '%20'.join(group)
        coher-=requests.get(url=url).json()'''
    '''url = get_str + '%20'.join(word_list[:10])
    coher = requests.get(url=url).json()
    print(f'Topic Coherence of {id2topic[topic]}: {str(coher)}')
    return coher'''
    return word_list

def get_tfidf(data_dir, word2id):
    vocab_freq = np.zeros((len(word2id)))
    df = np.zeros((len(word2id)))
    with open(os.path.join(data_dir, 'tf.txt')) as f:
        for line in f:
            w = line.split(' ')[0]
            f = line.strip().split(' ')[2]
            try:
                vocab_freq[word2id[w]] = int(f)
            except KeyError:
                continue
    with open(os.path.join(data_dir, 'df.txt')) as f:
        for line in f:
            w = line.split(' ')[0]
            f = line.strip().split(' ')[1]
            try:
                df[word2id[w]] = int(f)
            except KeyError:
                continue

    return vocab_freq, df


def rank_cap(cap, idx2word, class_idx):
    target_cap = np.mean(np.array([cap[x] for x in class_idx]))
    print(target_cap)
    word_cap = np.zeros(len(idx2word))
    for i in range(len(idx2word)):
        if idx2word[i] in cap:
            word_cap[i] = (cap[idx2word[i]]-target_cap) ** 2
        else:
            word_cap[i] = np.array([1.0])

    low2high = np.argsort(word_cap)
    #print(low2high[class_idx])
    #class_idx_rank = low2high[class_idx]
    #low2high = [np.inf if x < class_idx_rank else int(x - class_idx_rank + 1) for x in low2high]
    #print(low2high[class_idx])
    # mark = np.where(low2high==class_idx)[0]
    # assert(len(mark)) == 1
    # high2low = np.argsort(word_cap)
    # f = open("temp_cap.txt", "w")
    # for i in range(len(cap)):
    #     idx = low2high[i]
    #     f.write(idx2word[idx] + ' ' + str(word_cap[idx]) + '\n')
    return low2high,word_cap#[max(0,mark[0]-1000):min(mark[0]+1000,len(low2high)-1)]-mark[0]+1001

 
def aggregate_ranking(sim, cap, word_cap, topic, id2topic, idx2word, pretrain, out_file):
    # sim_cand = sim[:100]
    # cap_cand = cap[:100]
    simrank2id = np.ones(len(sim)) * np.inf
    caprank2id = np.ones(len(sim)) * np.inf
    for i, w in enumerate(sim[:200]):
        simrank2id[w] = i + 1
    for i, w in enumerate(cap):
        if pretrain == '0':
            # if word_cap[w] > word_cap_orig[topic]:
            caprank2id[w] = i + 1
    if pretrain == '0':
        agg_rank = simrank2id #* caprank2id
        final_rank = np.argsort(agg_rank)
        final_rank_words = [idx2word[idx] for idx in final_rank[:50]]
    else:
        agg_rank = simrank2id
        final_rank = np.argsort(agg_rank)
        final_rank_words = [idx2word[idx] for idx in final_rank[:50]]
    print(f'\n{topic} ranking list:')
    print([caprank2id[idx] for idx in sim[:20]])
    print([simrank2id[idx] for idx in final_rank[:20]])
    print([caprank2id[idx] for idx in final_rank[:20]])
    print(final_rank_words)
    f = open(out_file, 'a')
    # f.write(f'\n{topic}:\n')
    f.write(' '.join(final_rank_words) + '\n')
    # print([agg_rank[idx] for idx in final_rank[:20]])
    return

def read_file(file_name, word2id, outfile):
    f = open(file_name)
    docs = f.readlines()
    docs = [doc.strip().split(' ') for doc in docs]
    corpus = []
    for doc in docs:
        document = []
        for w in doc:
            if w in word2id:
                document.append(word2id[w])
        corpus.append(document)
    pickle.dump(corpus, open(outfile, 'wb'))


def make_emb_mat(word_emb, vocabulary_inv):
    embedding = np.zeros((len(word_emb), 100))
    for i in range(len(word_emb)):
        embedding[i] = word_emb[vocabulary_inv[i]]
    return embedding



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='20news')
    parser.add_argument('--emb', default='emb_an_final')
    parser.add_argument('--topic_file', default='topics.txt')
    parser.add_argument('--out_file', default='results.txt')
    parser.add_argument('--test', default='local',choices=['local','global'])
    parser.add_argument('--pretrain', default='0', help='pretrained:1, else:0')
    parser.add_argument('--exp', default='1.0',type=float)

    args = parser.parse_args()
    print(args)

    topic_prefix = args.topic_file.split('topics_')[1].split('.txt')[0]

    topic2id = {}
    id2topic = {}
    idx=0
    with open(os.path.join(args.dataset, args.topic_file)) as f:
        for line in f:
            id2topic[str(idx)] = line.strip()
            topic2id[line.strip()] = str(idx)
            idx += 1
    print(id2topic)

    if args.pretrain == '0':

        
        word_emb, vocabulary, vocabulary_inv = get_emb(vec_file=os.path.join(args.dataset, args.emb + '_w.txt'))

        #wemb = np.array(filtered_vectors(vocabulary_inv, 100))
        topic_emb, _, _ = get_emb(vec_file=os.path.join(args.dataset, args.emb + '_t.txt'))

        # vocab_freq, df = get_tfidf(args.dataset,vocabulary)
        word_cap = get_cap(vec_file=os.path.join(args.dataset, args.emb + '_cap.txt'))
        word_cap_orig=word_cap
        # rank_cap(word_cap, vocabulary_inv, 20)
        #print(word_emb['8086'])

        #if the word embedding is matrix instead of dictionary
        #for idx in vocabulary_inv:
        #    word_emb[vocabulary_inv[idx]] = wemb[idx]
    else:
        #word_emb, vocabulary, vocabulary_inv = get_emb(vec_file=os.path.join(args.dataset, args.emb + '_w.txt' ))
        word_emb, vocabulary, vocabulary_inv = get_emb(vec_file=os.path.join(args.dataset, args.emb ))
        # word_cap = get_cap(vec_file=os.path.join(args.dataset, args.emb.replace('vec', 'cap')))
        # word_cap = get_cap(vec_file=os.path.join(args.dataset, args.emb + '_cap.txt'))
        # print(word_cap['france'])
    print(word_cap_orig['germany'])


    

    #if there is no explicit topic embedding
    if args.pretrain == '1':
        print('pretrained embedding')
        topic_emb = {}
        for topic in topic2id:
            topic_emb[topic2id[topic]] = word_emb[topic]

    #seed_topic_distr(id2topic, topic_emb, word_emb)

    # t1 = 'earning'
    # w2 = 'earning'
    # t2 = 'grain'
    # #print(topic_emb['0'])
    # print(f'similarity of words {t1} and {w2} : {calc_sim(word_emb,(t1,w2))}')
    # print(f'similarity of topic {t1} and word {w2} : {calc_sim2(topic_emb, word_emb,(topic2id[t1],w2))}')
    # print(f'similarity of topic {t1} and topic {t2} : {calc_sim(topic_emb,(topic2id[t1],topic2id[t2]))}')


    #for topic in topic_emb:
    #    most_sim(topic, id2topic, vocabulary_inv, topic_emb, word_emb, 15)


    #print(id2topic)
    #print(topic2id)

    word_list = []

    f = open(os.path.join('../result', args.dataset.split('../')[1], topic_prefix, args.out_file), 'w')
    f.close()

    for topic in topic2id:
        sim_ranking = topic_sim(topic, vocabulary_inv, topic_emb, word_emb)
        if args.pretrain == '1':
            cap_ranking = np.ones((len(vocabulary)))
            word_cap1 = np.ones((len(vocabulary)))
        else:
            print(topic)
            cap_ranking, word_cap1  = rank_cap(word_cap, vocabulary_inv, [x for x in topic.split(' ')])
        aggregate_ranking(sim_ranking, cap_ranking, word_cap1, topic, id2topic, vocabulary_inv, args.pretrain, os.path.join('../result', args.dataset.split('../')[1], topic_prefix, args.out_file))
        # word_list.append(topic_coherence(topic, id2topic, vocabulary_inv, topic_emb, word_emb, 20, args.test, \
        #                                  word_cap, 20))
        # print(word_list[int(topic)])

    # for topic in topic_emb:
    #     print(f'{topic}: {word_cap[topic]}')






