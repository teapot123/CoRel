import gensim
from gensim.models import Word2Vec, KeyedVectors
import mmap
from tqdm import tqdm
import json
import re
import logging
import sys
from bert_serving.client import BertClient
import numpy as np
import os
import torch
from pytorch_transformers import *
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def processOneLine(line):
    """Return a (list of) sentence(s) with entity id replaced."""
    line = line.strip()
    tmp = line.split('<phrase>')
    if len(tmp) < 2:
        return 0,0
    else:
        ent_list = []
        context_list = []
        for seg in tmp:
            temp2 = seg.split('</phrase>')
            if len(temp2) < 2:
                continue
            ent = temp2[0]
            phrase_ent = '<phrase>'+ent+'</phrase>'
            sentence = line.replace(phrase_ent, '[MASK]')
            sentence = sentence.replace('<phrase>', '')
            sentence = sentence.replace('</phrase>', '')
            # print(ent)
            # print(sentence)
            if ent not in ent_freq:
                ent_freq[ent] = 0
            ent_freq[ent] += 1

            ent_list.append(ent)
            sentence = "[CLS] "+sentence+" [SEP]"
            context_list.append(sentence)
        return ent_list, context_list


def extract_entity_embed_and_save(model, output_file, output_file2):
    
    model_size = 768
    vocab_size = len(model)
    print("Saving embedding: model_size=%s,vocab_size=%s" % (model_size, vocab_size))
    with open(output_file, 'w') as f, open(output_file2,'w') as f2:
        for eid in model:
            f.write("{} {}\n".format(eid, ' '.join([str(x) for x in model[eid]])))
            f2.write("{} {}\n".format(eid, ent_freq[eid]))


if __name__ == "__main__":
    corpusName = sys.argv[1]
    num_thread = int(sys.argv[2])
    inputFilePath = corpusName+"/segmentation.txt"
    saveFilePath = corpusName+"/BERTembed.txt"
    saveFilePath2 = corpusName+"/BERTembednum.txt"

    sentences = []
    bert_embedding = {}
    ent_freq = {}
    word_type_embedding = {}
    ent_record = []
    window_size = 5

    print(torch.cuda.device_count())


    model_class = BertModel
    tokenizer_class = BertTokenizer
    pretrained_weights = 'bert-base-uncased'
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



    with open(inputFilePath, "r") as fin:
        tmp = fin.readlines()
        for line in tqdm(tmp, total=len(tmp), desc="loading corpus for retrieving bert embedding"):
            ent,ctxt = processOneLine(line)
            if ent == 0:
                continue
            sentences.extend(ctxt)
            ent_record.extend(ent)
            if len(sentences) > 50:
                tokens = [tokenizer.encode(x) for x in sentences]
                # print(f"length of tokens: {len(tokens)}")
                # print(f"length of entities: {len(ent_record)}")
                # print(torch.cat([F.pad(torch.tensor(x).unsqueeze(),(0,60-len(x)), "constant", 0) for x in tokens]))
                # input_ids = torch.cat([F.pad(torch.tensor(x).unsqueeze(0),(0,4*window_size+3-len(x)), "constant", 0) for x in tokens],dim=0).cuda()
                input_ids = torch.cat([F.pad(torch.tensor(x).unsqueeze(0),(0,24-len(x)), "constant", 0) for x in tokens],dim=0).cuda()

                # if len(sentences) > 20:
                #     print(f'batch size {len(sentences)} too large')
                #     sentences = []
                #     ent_record = []
                #     continue
                # print(input_ids.shape)
                vec = model(input_ids)[0]
                # print(vec.shape)
                for i,e in enumerate(ent_record):
                    try:
                        mask_id = tokens[i].index(103)
                        vec0 = vec[i][mask_id]
                        # print(vec0)
                        if len(vec0)!=768:
                            print(f"length = {len(vec0)}")
                            break
                        if e not in bert_embedding:
                            bert_embedding[e] = []
                        bert_embedding[e].append(vec0.cpu().detach().numpy())
                    # except TypeError:
                    #     print(i)
                    #     print(e)
                    #     print(vec)
                    # except ValueError:
                    #     print(tokenizer.tokenize(sentences[i]))
                    except IndexError:
                        pass
                        # print(tokenizer.tokenize(sentences[i]))
                        # print(e)
                        # print(vec0)
                        # print(vec[i].shape)
                        # print(mask_id)
                sentences = []
                ent_record = []

    print('finish bert embedding calculation')

    for eid in tqdm(bert_embedding,total=len(bert_embedding)):
        word_type_embedding[eid] = np.mean(np.array(bert_embedding[eid]),axis=0)

    extract_entity_embed_and_save(word_type_embedding, saveFilePath, saveFilePath2)
