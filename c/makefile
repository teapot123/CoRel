CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: train_emb_full_tax train_emb_part_tax 


train_emb_full_tax : train_emb_full_tax.c
	$(CC) train_emb_full_tax.c -o train_emb_full_tax $(CFLAGS)
train_emb_parttax : train_emb_parttax.c
	$(CC) train_emb_part_tax.c -o train_emb_part_tax $(CFLAGS)
clean:
	rm -rf train_emb_full_tax 
	rm -rf train_emb_part_tax  
