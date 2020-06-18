import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='dblp')
    parser.add_argument('--keyword_file', default='keyword_taxonomy.txt')
    parser.add_argument('--firstlayer_file', default='topics_firstlayer.txt')
    parser.add_argument('--out_file', default='run_emb_full_tax.sh')
    parser.add_argument('--template', default='template.sh')

    args = parser.parse_args()
    print(args)
    dataset = args.dataset
    keyword_file = args.keyword_file

    first_layer_topics = []
    first_layer = 1
    with open(os.path.join(dataset, keyword_file)) as f:
    	for line in f:
    		if len(line.strip()) == 0:
    			first_layer = 1
    			continue
    		if first_layer:
    			first_layer_topics.append(line.strip())
    			first_layer = 0

    with open(os.path.join(dataset, args.firstlayer_file),'w') as fout:
        for t in first_layer_topics:
            fout.write(t+'\n')

    first_layer_topics.append('firstlayer')

    with open(os.path.join('c', args.template)) as f:
	    with open(os.path.join('c', args.out_file), 'w') as fout:
	    	for line in f:
	    		if line.strip().find('dataset=../') == 0:
	    			fout.write('dataset='+dataset+'\n')
	    		elif line.strip().find('for subtopic in ') == 0:
	    			fout.write('for subtopic in '+' '.join(first_layer_topics)+'\n')
	    		else:
	    			fout.write(line)



