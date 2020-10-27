import argparse

if __name__=="__main__":
	
	parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--multi', default='0.5')
	parser.add_argument('--single', default = '0.8')
	parser.add_argument('--mode', default='whole', choices=['phrase', 'whole'])
	parser.add_argument('--output',default='20news')
	args = parser.parse_args()
	file_name = 'models/'+args.output+'/phrase_dataset_'+str(args.multi)+'_'+str(args.single)+'.txt'
	#if args.mode == 'whole':
	#	file_name = 'models/new_nyt/dataset_'+str(args.multi)+'_'+str(args.single)+'.txt'
	#with open('../out.txt') as f:
	phrases = {}
	with open('models/'+args.output+'/AutoPhrase_multi-words.txt') as f:
		for line in f:
			if len(line.strip().split('\t'))<2:
				print(line)
			phrases[line.strip().split('\t')[1]] = line.split('\t')[0]
	with open('models/'+args.output+'/segmentation.txt') as f:
		with open(file_name, 'w') as g:
			i = 0
			word_count = 0
			for line in f:
				doc = ''
				i += 1
				#if i % 1000 == 0:
					#print(i)
				temp = line.split('<phrase>')
				if args.mode == 'phrase':
					for seg in temp:
						temp2 = seg.split('</phrase>')
						if len(temp2) > 1:
							doc += ('_').join(temp2[0].split(' ')) + ' '
							#doc += temp2[0] + ' '
					word_count += len(doc.split(' '))
					g.write(doc+'\n')
				else:
					for seg in temp:
						temp2 = seg.split('</phrase>')
						if len(temp2) > 1:
							doc += ('_').join(temp2[0].split(' ')) + temp2[1]
							# if temp2[0] not in phrases:
							# 	doc += temp2[0] + temp2[1]
							# else:
							# 	doc += ('_').join(temp2[0].split(' ')) + temp2[1]
						else:
							doc += temp2[0]
					word_count += len(doc.split(' '))
					g.write(doc.strip()+'\n')
			print(word_count/i)
				