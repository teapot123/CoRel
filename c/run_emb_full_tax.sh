# dataset directory
dataset=dblp

# text file name; one document per line
text_file=phrase_text.txt

mkdir ../result
rm -r ../result/${dataset}
mkdir ../result/${dataset}

for subtopic in data_mining firstlayer
do
  topic_file=${subtopic}

  output_file=${topic_file}

  emb_file=emb_${output_file}

  make train_emb_full_tax

  # default:
  # reg_lambda = 10
  # global_lambda = 1.5
  # window = 5
  # pretrain = 5

  echo ${green}=== Jointly Training Concept Embedding and Text Embedding===${reset}
  # ./train_emb_full_tax -train ../${dataset}/${text_file} -output ../${dataset}/${emb_file}_w.txt -kappa ../${dataset}/${emb_file}_cap.txt -topic ../${dataset}/topics_${topic_file}.txt -topic_output ../${dataset}/${emb_file}_t.txt -doc_output ../${dataset}/${emb_file}_s.txt -fix_seed 1 -reg_lambda 1 -cbow 0 -size 100 -global_lambda 1.5 -window 5 -negative 5 -sample 1e-3 -min-count 50 -threads 20 -binary 0 -iter 2 -pretrain 1 -rank_product 0 -gen_vocab 0 -load_emb 0
 
  echo ${green}=== Using Trained Embedding to Generate Keywords===${reset}
  mkdir ../result/${dataset}/${subtopic}
  python ../generate_final_taxonomy.py --dataset ../${dataset} --topic_file topics_${topic_file}.txt --emb ${emb_file} --out_file subtopics_${output_file}.txt

done
