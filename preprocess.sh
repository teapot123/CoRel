#!/bin/bash

dataset=yelp
num_thread=20
multi=0.75
single=0.65

# ----Run This Before Setting Autophrase Threshold-----
python extractCorpus.py ${dataset} ${num_thread}
cd AutoPhrase0/
bash auto_phrase_cp.sh ${dataset}

# ----Run This After Setting Autophrase Threshold-----
cd AutoPhrase/
bash phrasal_segmentation_cp.sh ${dataset} corpus.txt ${multi} ${single}
cp models/${dataset}/phrase_dataset_${multi}_${single}.txt ../${dataset}/phrase_text.txt
bash phrasal_segmentation_cp.sh ${dataset} sentences.txt ${multi} ${single}
cp models/${dataset}/segmentation.txt ../${dataset}/
cd ../
python extractSegmentation.py ${dataset}
python extractBertEmbedding.py ${dataset} 20
