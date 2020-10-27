import gensim
from gensim.models import Word2Vec, KeyedVectors
import mmap
from tqdm import tqdm
import json
import re
import logging
import sys
from collections import defaultdict


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def minDuplicate(intervals):
    starts = []
    ends = []
    for i in intervals:
        starts.append(i[0])
        ends.append(i[1])

    starts.sort()
    ends.sort()
    s = e = 0
    numDuplicate = available = 0
    while s < len(starts):
        if starts[s] <= ends[e]:  # when an entity span starts, the previous entity span does not end
            if available == 0:   # if an available sentence sequence doesn't exit
                numDuplicate += 1
            else:
                available -= 1
            s += 1
        else:  # a new entity span starts after the previous entity span ends
            available += 1
            e += 1

    return numDuplicate

def countWords(sentInfo):
    tokens = sentInfo['tokens']
    return len(tokens)


def processOneLine(sentInfo):
    """Return a (list of) sentence(s) with entity id replaced."""
    tokens = sentInfo["tokens"]
    return tokens, sentInfo["articleId"]

def processOneLineWithEntity(line):
    """Return a (list of) sentence(s) with entity id replaced."""
    tmp = line.split('<phrase>')
    if len(tmp) <= 2:
        return None, None
    entityMentions = []
    sentence = ''
    for seg in tmp:
        temp2 = seg.split('</phrase>')
        if (len(temp2) > 1):
            entityMentions.append(('_').join(temp2[0].split(' ')))
            sentence += ('_').join(temp2[0].split(' ')) + temp2[1]
        else:
            sentence += temp2[0]
    return sentence, entityMentions 


def trim_rule(word, count, min_count):
    """Used in word2vec model to make sure entity tokens are preserved. """
    if re.match(r"^entity\d+$", word):  # keep entity token
        return gensim.utils.RULE_KEEP
    else:
        return gensim.utils.RULE_DEFAULT


def extract_entity_embed_and_save(model, output_file):
    def match_rule(word):
        if re.match(r"^entity\d+$", word):
            return True
        else:
            return False

    model_size = model.vector_size
    vocab_size = len([word for word in model.wv.vocab if match_rule(word)])
    print("Saving embedding: model_size=%s,vocab_size=%s" % (model_size, vocab_size))
    with open(output_file, 'w') as f:
        for word in model.wv.vocab:
            if match_rule(word):
                vector_string = " ".join([str(ele) for ele in list(model.wv[word])])
                f.write("{} {}\n".format(word[6:], vector_string))


if __name__ == "__main__":
    corpusName = sys.argv[1]
    num_thread = 20

    neg = 5
    ws = 10

    inputFilePath = corpusName+"/segmentation.txt"

    sentences = defaultdict(list)
    ent_sent_dict = defaultdict(list)
    ent_ent_dict = defaultdict(set)
    with open(inputFilePath, "r") as fin:
        count = 0
        for line in tqdm(fin, total=get_num_lines(inputFilePath), desc="loading corpus for word2vec training"):
            sentence, entities = processOneLineWithEntity(line)
            if sentence == None:
                continue
            sentences[count] = sentence
            if len(entities) > 1:
                for ent in entities:
                    ent_sent_dict[ent].append(str(count))
                for i, e1 in enumerate(entities):
                    for j, e2 in enumerate(entities):
                        if j==i or e1==e2:
                            continue
                        ent_ent_dict[e1].add(e2)
            count += 1
            

    # write sentences.txt and ent_sent_index.txt
    with open(corpusName+"/sentences_.txt", "w") as fout: 
        for line in tqdm(range(count), total=len(sentences)):
            fout.write(sentences[line])

    with open(corpusName+"/ent_sent_index.txt", "w") as fout:
        for ent in tqdm(ent_sent_dict, total=len(ent_sent_dict)):
            fout.write(ent+"\t"+" ".join(ent_sent_dict[ent])+"\n")

    with open(corpusName+"/ent_ent_index.txt", "w") as fout:
        for ent in tqdm(ent_ent_dict, total=len(ent_ent_dict)):
            fout.write(ent+"\t"+" ".join(ent_ent_dict[ent])+"\n")

