# CoRel: Seed-Guided Topical Taxonomy Construction by Concept Learning and Relation Transferring

The source code used for our KDD'2020 paper. 

## Requirements

* GCC compiler (used to compile the source c file): See the [guide for installing GCC](https://gcc.gnu.org/wiki/InstallingGCC).

## Datasets 

Due to the constraint of size, we provide the link of our datasets in the following links, please copy the files to ``${dataset}/``. 
* [DBLP & Yelp](https://drive.google.com/drive/folders/1t9IrrLm1fB92IC2nwGk-mIJCNubzx2fT?usp=sharing)

## Run the Code

### Concept Learning for topic nodes
```
cd c
bash run_emb_part_tax.sh
```
This step compiles the source file and trains embedding for concept learning. The ``--topic_file`` in the script is used to specify the seed taxonomy. 
As an example, you can set ${topic_file} to be ``topics_field.txt`` for dataset ``dblp`` and ``topics_des.txt`` for dataset ``yelp``. These topic files are already provided in the datasets. If you want to specify your own seed taxonomy, just feel free to create a new file using the format ``topics_{xxx}.txt``.
Each line starts with a parent node (with the root node being ROOT), and then followed by a ``tab``. The children nodes of this parent is appended and separated by ``space``. Generated embedding file is stored under ``${dataset}``.

### Relation Transferring
A jupyter notebook version is available for this step.
You can change the dataset and the topic_file name in ``main.ipynb``.
As another option, you can use the following python programme to generate the results.
```
cd ..
python main.py --dataset ${dataset} --topic_file ${topic_file}.txt
```
This step completes the taxonomy structure and outputs keywords for each node in the taxonomy. 
As an example, for DBLP dataset, you can run
```
python main.py --dataset dblp --topic_file topics_field.txt
```
### Concept learning for all nodes
```
python generate_bash.py
```
This command generates a script from ``c/template.sh`` that can recursively run embedding training for all topics and subtopics.
```
cd c
bash run_emb_full_tax.sh
```
This command will generate the final topical taxonomy under a ``result`` directory.

## Results
Results for each topics are generated at ``result\${dataset}\${topic}\subtopics_for_${topic}.txt``.
E.g., each line in ``result\DBLP\data_mining\subtopics_for_data_mining.txt`` is one subtopic of data mining (including a cluster of words).
