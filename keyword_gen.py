import numpy as np
import pickle
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
    return low2high,word_cap
 
def aggregate_ranking(sim, cap, word_cap, topic, id2topic, idx2word, pretrain, out_file):
    simrank2id = np.ones(len(sim)) * np.inf
    caprank2id = np.ones(len(sim)) * np.inf
    for i, w in enumerate(sim[:200]):
        simrank2id[w] = i + 1
    for i, w in enumerate(cap):
        # if word_cap[w] > word_cap_orig[topic]:
        caprank2id[w] = i + 1
    agg_rank = simrank2id #* caprank2id
    final_rank = np.argsort(agg_rank)
    final_rank_words = [idx2word[idx] for idx in final_rank[:50]]
    print(f'\n{topic} ranking list:')
    print([caprank2id[idx] for idx in sim[:20]])
    print([simrank2id[idx] for idx in final_rank[:20]])
    print([caprank2id[idx] for idx in final_rank[:20]])
    print(final_rank_words)
    f = open(out_file, 'a')
    f.write(f'\n{topic}:\n')
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
    parser.add_argument('--dataset', default='dblp')
    parser.add_argument('--emb', default='')
    parser.add_argument('--topic_file', default='topics_field.txt')
    parser.add_argument('--out_file', default='results.txt')
    parser.add_argument('--test', default='local',choices=['local','global'])
    parser.add_argument('--pretrain', default='0', help='pretrained:1, else:0')
    parser.add_argument('--exp', default='1.0',type=float)

    args = parser.parse_args()
    print(args)

    topic2id = {}
    id2topic = {}
    idx=0
    with open(os.path.join(args.dataset, args.topic_file)) as f:
        for line in f:
            id2topic[str(idx)] = line.strip()
            topic2id[line.strip()] = str(idx)
            idx += 1
    print(id2topic)
      
    word_emb, vocabulary, vocabulary_inv = get_emb(vec_file=os.path.join(args.dataset, args.emb + '_w.txt'))
    topic_emb, _, _ = get_emb(vec_file=os.path.join(args.dataset, args.emb + '_t.txt'))
    word_cap = get_cap(vec_file=os.path.join(args.dataset, args.emb + '_cap.txt'))
    word_cap_orig=word_cap
        
    word_list = []

    f = open(os.path.join(args.dataset, args.out_file), 'w')
    f.close()

    for topic in topic2id:
        sim_ranking = topic_sim(topic, vocabulary_inv, topic_emb, word_emb)
        if args.pretrain == '1':
            cap_ranking = np.ones((len(vocabulary)))
            word_cap1 = np.ones((len(vocabulary)))
        else:
            print(topic)
            cap_ranking, word_cap1  = rank_cap(word_cap, vocabulary_inv, [x for x in topic.split(' ')])
        aggregate_ranking(sim_ranking, cap_ranking, word_cap1, topic, id2topic, vocabulary_inv, args.pretrain, os.path.join(args.dataset, args.out_file))




