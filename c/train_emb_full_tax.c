//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>


#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

// windows mingw
//#define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ?0 :errno)

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int corpus_max_size = 30000000;  // Maximum 30M documents in the corpus

typedef float real;                    // Precision of float numbers

struct vocab_word {
    long long cn;
    int *point;
    char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING], output_tmp_file[MAX_STRING], kappa_file[MAX_STRING], topic_file[MAX_STRING];
char topic_output[MAX_STRING];
char context_output[MAX_STRING];
char doc_output[MAX_STRING];
char load_emb_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash, *docs;
long long *doc_sizes;
long long vocab_max_size = 1000, vocab_size = 0, corpus_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 10, file_size = 0, classes = 0;
int pretrain_iters = 2;
int is_pretrain = 1; //
real alpha = 0.025, starting_alpha, global_lambda = 1.5, sample = 1e-3;
real *syn0, *syn1neg, *syn1doc, *expTable;
real *kappa;
clock_t start;

int with_global = 1;
int with_regularization = 0;
int with_kappa = 0;

int negative = 5;
const int table_size = 1e8;
int *word_table, *doc_table;

// regularization:
int topics = 0; // number of topics, t_embedding.size(0)
int topic_words = 0;
int *topic_index;
int *topic_start_end;
int multi_keywords = -1;

real *t_embeddings;
int initial_seed_set_size = 1;
int expand = 1; // iter * expand + initial_seed_set_size == rankings.size()
int num_per_topic;
real *wt_score; // similarity between topic and word
int *rankings; // ranking for each topic
int *rankings1; // ranking for each topic
int *fix_seed_rankings; // fixed the seed topics in the front rankings
real reg_lambda = 1.0; // scaling for regularization
int rank_product = 0;

int topic_pivot_idx;
int similaritySearchSize;
int *sim_rankings;
int *kappa_rankings;

int gen_vocab = 0;
int load_emb = 0;
int load_emb_with_v = 0;
int fix_seed = 0;


int simRankingComparator(const void *a, const void *b) { // large -> small
  return (wt_score[*(int *) a] < wt_score[*(int *) b]) - (wt_score[*(int *) a] > wt_score[*(int *) b]);
}

int kappaRankingComparator(const void *a, const void *b) { // small -> large
  real aSmall0 = (kappa[(*(int *) a) % vocab_size] - kappa[topic_pivot_idx]) * (kappa[(*(int *) a) % vocab_size] - kappa[topic_pivot_idx]);
  real bSmall0 = (kappa[(*(int *) b) % vocab_size] - kappa[topic_pivot_idx]) * (kappa[(*(int *) b) % vocab_size] - kappa[topic_pivot_idx]);
  int aSmall = aSmall0 > bSmall0;
  int bSmall = bSmall0 > aSmall0;
  return aSmall - bSmall;
}

int productRankingComparator(const void *a, const void *b) { // small -> large
  long long rankA = 1LL * kappa_rankings[(*(int *) a)] * sim_rankings[*(int *) a];
  long long rankB = 1LL * kappa_rankings[(*(int *) b)] * sim_rankings[*(int *) b];
  if (sim_rankings[*(int *) a] > similaritySearchSize || kappa[(*(int *) a)] ) {
    rankA = 1e18;
  }
  if (sim_rankings[*(int *) b] > similaritySearchSize || kappa[(*(int *) b)] ) {
    rankB = 1e18;
  }
  // if (kappa[(*(int *) a) % vocab_size] < kappa[topic_pivot_idx]) {
  //   rankA = 1e18;
  // }
  // if (kappa[(*(int *) b) % vocab_size] < kappa[topic_pivot_idx]) {
  //   rankB = 1e18;
  // }
  return (rankA > rankB) - (rankA < rankB);
}


void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  word_table = (int *) malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    word_table[a] = i;
    if (a / (double) table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

void InitDocTable() {
  int a, i;
  double doc_len_pow = 0;
  double d1, power = 0.75;
  doc_table = (int *) malloc(table_size * sizeof(int));
  for (a = 0; a < corpus_size; a++) doc_len_pow += pow(docs[a], power);
  i = 0;
  d1 = pow(docs[i], power) / doc_len_pow;
  for (a = 0; a < table_size; a++) {
    doc_table[a] = i;
    if (a / (double) table_size > d1) {
      i++;
      d1 += pow(docs[i], power) / doc_len_pow;
    }
    if (i >= corpus_size) i = corpus_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *) "</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Record document length
void ReadDoc(FILE *fin) {
  char word[MAX_STRING];
  long long i;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    i = SearchVocab(word);
    if (i == 0) {
      doc_sizes[corpus_size] = ftell(fin);
      corpus_size++;
    } else if (i == -1) continue;
    else docs[corpus_size]++;
  }
  // for (i = 0; i <= 5; i++) printf("%lld\n", doc_sizes[i]);
}

// Locate line number of current file pointer
int FindLine(FILE *fin) {
  long long pos = ftell(fin);
  long long lo = 0, hi = corpus_size - 1;
  while (lo < hi) {
    long long mid = lo + (hi - lo) / 2;
    if (doc_sizes[mid] > pos) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) { // assert all sortings will be the same (since c++ qsort is not stable..)
  if (((struct vocab_word *) b)->cn == ((struct vocab_word *) a)->cn) {
    return strcmp(((struct vocab_word *) b)->word, ((struct vocab_word *) a)->word);
  }
  return ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *) realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *) calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *) calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++)
    if (vocab[a].cn > min_reduce) {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      b++;
    } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *) "</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fseek(fin, 0, SEEK_SET);
  ReadDoc(fin);
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **) &syn0, 128, (long long) vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }
  a = posix_memalign((void **) &kappa, 128, (long long) vocab_size * sizeof(real));
  for (a = 0; a < vocab_size; a++) kappa[a] = 1.0;
  if (negative > 0) {
    a = posix_memalign((void **) &syn1neg, 128, (long long) vocab_size * layer1_size * sizeof(real));
    a = posix_memalign((void **) &syn1doc, 128, (long long) corpus_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {
      printf("Memory allocation failed (syn1neg)\n");
      exit(1);
    }
    if (syn1doc == NULL) {
      printf("Memory allocation failed (syn1doc)\n");
      exit(1);
    }
    for (a = 0; a < vocab_size; a++)
      for (b = 0; b < layer1_size; b++)
        syn1neg[a * layer1_size + b] = 0;
    for (a = 0; a < corpus_size; a++)
      for (b = 0; b < layer1_size; b++)
        syn1doc[a * layer1_size + b] = 0;
  }
  if (gen_vocab) {
    FILE *fp = fopen("vocabs.txt", "w");
    fprintf(fp, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fp, "%s\n", vocab[a].word);
    }
    fclose(fp);
    printf("Just generating the vocabulary, exiting\n");
    exit(0);
  }
  if (load_emb && load_emb_with_v) {
    printf("Duplicate pretrained embedding! Please set load_emb or load_emb_with_v to 1.\n");
    exit(0);
  }
  if (load_emb) {
    printf("loading embedding from emb_w.txt, emb_v.txt\n");
    if (access("emb_w.txt", R_OK) == -1) {
      printf("emb_w.txt does not exist\n");
      return;
    }
    if (access("emb_v.txt", R_OK) == -1) {
      printf("emb_v.txt does not exist\n");
      return;
    }

    FILE *fp = fopen("emb_w.txt", "r");
    for (a = 0; a < vocab_size; a++)
      for (b = 0; b < layer1_size; b++)
        fscanf(fp, "%f", &syn0[a * layer1_size + b]);
    fclose(fp);
    fp = fopen("emb_v.txt", "r");
    for (a = 0; a < vocab_size; a++)
      for (b = 0; b < layer1_size; b++)
        fscanf(fp, "%f", &syn1neg[a * layer1_size + b]);
    fclose(fp);
  } else if(load_emb_with_v){
    char * center_emb_file = (char *) calloc(MAX_STRING, sizeof(char));
    char * context_emb_file = (char *) calloc(MAX_STRING, sizeof(char));
    if(load_emb_file[0] == 0) {
      strcpy(center_emb_file, "emb_w.txt");
      strcpy(context_emb_file, "emb_v.txt");
    } else {
      strcpy(center_emb_file, load_emb_file);
      strcat(center_emb_file, "_w.txt");
      strcpy(context_emb_file, load_emb_file);
      strcat(context_emb_file, "_v.txt");
    }
    printf("loading embedding from %s, %s\n", center_emb_file, context_emb_file);
    if (access(center_emb_file, R_OK) == -1) {
      printf("%s does not exist\n",center_emb_file);
      return;
    }
    if (access(context_emb_file, R_OK) == -1) {
      printf("%s does not exist\n",context_emb_file);
      return;
    }
    int vocab_size_tmp, word_dim;
    char * current_word = (char *) calloc(MAX_STRING, sizeof(char));
    FILE *fp = fopen(center_emb_file, "r");
    fscanf(fp, "%d", &vocab_size_tmp);
    fscanf(fp, "%d", &word_dim);
    for (a = 0; a < vocab_size; a++) {
      fscanf(fp, "%s", current_word);
      if(strcmp(current_word, vocab[a].word) != 0) {
        printf("vocabulary is not aligned! Line %lld should be %s\n",a+2,vocab[a].word);
        exit(0);
      }
      for (b = 0; b < layer1_size; b++)
        fscanf(fp, "%f", &syn0[a * layer1_size + b]);
    }
    fclose(fp);
    fp = fopen(context_emb_file, "r");
    fscanf(fp, "%d", &vocab_size_tmp);
    fscanf(fp, "%d", &word_dim);
    for (a = 0; a < vocab_size; a++) {
      fscanf(fp, "%s", current_word);
      if(strcmp(current_word, vocab[a].word) != 0) {
        printf("vocabulary is not aligned! Line %lld should be %s\n",a+2,vocab[a].word);
        exit(0);
      }
      for (b = 0; b < layer1_size; b++)
        fscanf(fp, "%f", &syn1neg[a * layer1_size + b]);
    }
    fclose(fp);
  } else{
    for (a = 0; a < vocab_size; a++)
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
      }
  }
  // regularization
  if (with_regularization) {
    int iidx = 0, vocab_idx, lineIdx = 0;
    char tmp_word[MAX_STRING];
    memset(tmp_word, '\0', sizeof(tmp_word));
    for (a = 0; a < 2; a++) {
      FILE *fp = fopen(topic_file, "r");
      char *line = NULL;
      size_t len = 0;
      ssize_t read;
      while ((read = getline(&line, &len, fp)) != -1) {
        line[read - 1] = 0;
        if (line[read - 2] == '\r') // windows line ending
          line[read - 2] = 0;
        read = strlen(line);
        while (read > 0 && (line[read - 1] == ' ' || line[read - 1] == '\n')) read--;
        if (read == 0) {
          printf("[Topic] Empty String !\n");
          exit(1);
        }
        if (line[0] == ' ') {
          printf("[Topic] String starting with space!\n");
          exit(1);
        }
        for (int i = 0; i + 1 < read; i++) {
          if (line[i] == ' ' && line[i + 1] == ' ') {
            printf("[Topic] Consecutive spaces\n");
            exit(1);
          }
        }
        int st = 0;
        if (a != 0) {
          topic_start_end[2 * lineIdx] = iidx;
        }
        for (int i = 0; i <= read; i++) {
          if (line[i] == ' ' || i == read) {
            strncpy(tmp_word, line + st, i - st);
            tmp_word[i - st] = 0;
            if ((vocab_idx = SearchVocab(tmp_word)) != -1) {
              if (a != 0) {
                topic_index[iidx++] = vocab_idx;
              } else {
                topic_words++;
              }
            } else {
              printf("%s not found in vocab\n", tmp_word);
              exit(1);
            }
            st = i + 1;
          }
        }
        if (a == 0) topics++;
        if (a != 0) {
          topic_start_end[2 * lineIdx + 1] = iidx;
          lineIdx += 1;
        }
      }
      if (a == 0) {
        topic_index = calloc(topic_words, sizeof(int));
        topic_start_end = calloc(topics * 2, sizeof(int));
      }
      fclose(fp);
    }
    for (a = 0; a < topics; a++) {
      int words_in_topic = topic_start_end[a * 2 + 1] - topic_start_end[a * 2];
      if (a == 0) {
        initial_seed_set_size = words_in_topic;
      } else if (words_in_topic < initial_seed_set_size) {
        initial_seed_set_size = words_in_topic;
      }
      if (words_in_topic == 0) {
        printf("ERROR! \n");
        exit(1);
      }
      if (words_in_topic == 1) {
        if (multi_keywords == 1) {
          printf("Either multiple keywords, or one keyword per topic\n");
          exit(0);
        }
        multi_keywords = 2;
      } else {
        if (multi_keywords == 2) {
          printf("Either multiple keywords, or one keyword per topic\n");
          exit(0);
        }
        multi_keywords = 1;
      }
    }
    printf("Found %d topics\n", topics);
    for (a = 0; a < topics; a++) {
      for (int i = topic_start_end[a * 2]; i < topic_start_end[a * 2 + 1]; i++) {
        printf("%d %s   ", topic_index[i], vocab[topic_index[i]].word);
      }
      printf("\n");
    }
    a = posix_memalign((void **) &t_embeddings, 128, (long long) topics * layer1_size * sizeof(real));
    for (a = 0; a < topics; a++) {
      for (b = 0; b < layer1_size; b++) {
        t_embeddings[a * layer1_size + b] = 0.0;
      }
    }
    a = posix_memalign((void **) &wt_score, 128, (long long) topics * vocab_size * sizeof(real));
    a = posix_memalign((void **) &rankings, 128, (long long) topics * vocab_size * sizeof(int));
    a = posix_memalign((void **) &rankings1, 128, (long long) topics * vocab_size * sizeof(int));
    for (a = 0; a < topics * vocab_size; a++) rankings[a] = a;
    for (a = 0; a < topics * vocab_size; a++) rankings1[a] = a;
    a = posix_memalign((void **) &fix_seed_rankings, 128, (long long) topics * vocab_size * sizeof(int));
    if (rank_product) {
      a = posix_memalign((void **) &sim_rankings, 128, (long long) topics * vocab_size * sizeof(int));
      for (a = 0; a < topics * vocab_size; a++) sim_rankings[a] = a;
      a = posix_memalign((void **) &kappa_rankings, 128, (long long) topics * vocab_size * sizeof(int));
      for (a = 0; a < topics * vocab_size; a++) kappa_rankings[a] = a;
    }
  }
}

void *TrainModelThread(void *id) {
  long long a, b, d, doc, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = is_pretrain ? pretrain_iters : 1;
  unsigned long long next_random = (long long) id;
  real f, g;
  clock_t now;
  real *neu1 = (real *) calloc(layer1_size, sizeof(real));
  real *neu1e = (real *) calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);

  // regularization
  real *wi, *exp_rij, *delta_tjk, *delta_wik;
  int words_per_reg;
  int word_counter;
  if (with_regularization && !is_pretrain) {
    wi = (real *) calloc(topics * num_per_topic, sizeof(real));
    exp_rij = (real *) calloc(topics * num_per_topic * topics, sizeof(real));
    delta_tjk = (real *) calloc(topics * layer1_size, sizeof(real));
    delta_wik = (real *) calloc(topics * num_per_topic * layer1_size, sizeof(real));
    words_per_reg = 128;
    word_counter = 0;
  }
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now = clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
               word_count_actual / (real) (iter * train_words + 1) * 100,
               word_count_actual / ((real) (now - start + 1) / (real) CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real) (iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      doc = FindLine(fi);
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) /
                     vocab[word].cn;
          next_random = next_random * (unsigned long long) 25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real) 65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }

    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);
      continue;
    }
    // regularization
    if (with_regularization && !is_pretrain) {
      word_counter += 1;
      if (word_counter == words_per_reg) word_counter = 0;
      if (word_counter == 0) {
        real L = 0.0;
        for (a = 0; a < topics; a++)
          for (b = 0; b < num_per_topic; b++) {
            int word_index = fix_seed_rankings[a * vocab_size + b] % vocab_size;
            wi[a * num_per_topic + b] = 0;
            real rci = 0.0;

            for (d = 0; d < topics; d++) {
              real rij = 0.0;
              for (c = 0; c < layer1_size; c++) {
                rij += syn0[word_index * layer1_size + c] * t_embeddings[d * layer1_size + c];
              }
              real eij = exp(rij);
              exp_rij[(a * num_per_topic + b) * topics + d] = eij;
              wi[a * num_per_topic + b] += eij;
              if (a == d) rci = rij;
            }
            L += -rci + log(wi[a * num_per_topic + b]);
            for (c = 0; c < layer1_size; c++) {
              int wik_index = (a * num_per_topic + b) * layer1_size + c;
              delta_wik[wik_index] = 0.0;
              for (d = 0; d < topics; d++) {
                delta_wik[wik_index] += t_embeddings[d * layer1_size + c] *
                                        (exp_rij[(a * num_per_topic + b) * topics + d] /
                                         wi[a * num_per_topic + b] - (a == d ? 1 : 0));
              }
            }
          }
        for (d = 0; d < topics; d++)
          for (c = 0; c < layer1_size; c++) {
            int tjk_index = d * layer1_size + c;
            delta_tjk[tjk_index] = 0.0;
            for (a = 0; a < topics; a++)
              for (b = 0; b < num_per_topic; b++) {
                int word_index = fix_seed_rankings[a * vocab_size + b] % vocab_size;
                delta_tjk[tjk_index] += syn0[word_index * layer1_size + c] *
                                        (exp_rij[(a * num_per_topic + b) * topics + d] /
                                         wi[a * num_per_topic + b] - (a == d ? 1 : 0));
              }
          }
      }
    }

    word = sen[sentence_position];
    // printf("(%lld, %s)", doc, vocab[word].word);
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long) 25214903917 + 11;
    // b = next_random % window;
    b = 0;

    for (a = b; a < window * 2 + 1 - b; a++)
      if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // NEGATIVE SAMPLING
        if (negative > 0) {
          real kappa_update = 0.0;
          for (d = 0; d < negative + 1; d++) {
            if (d == 0) {
              target = word;
              label = 1;
            } else {
              next_random = next_random * (unsigned long long) 25214903917 + 11;
              target = word_table[(next_random >> 16) % table_size];
              if (target == 0) target = next_random % (vocab_size - 1) + 1;
              if (target == word) continue;
              label = 0;
            }
            l2 = target * layer1_size;
            f = 0;
            for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
            real tmp_kappa_update = f;
            f *= kappa[last_word];
            if (f > MAX_EXP) g = (label - 1) * alpha;
            else if (f < -MAX_EXP) g = (label - 0) * alpha;
            else g = (label - expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
            kappa_update += g * tmp_kappa_update;
            for (c = 0; c < layer1_size; c++) neu1e[c] += g * kappa[last_word] * syn1neg[c + l2];
            for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * kappa[last_word] * syn0[c + l1];
          }
          // Learn weights input -> hidden
          for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
          if (with_kappa) {
            kappa[last_word] += kappa_update;
          }
        }
      }
    if (with_global) {
      for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
      if (negative > 0) {
        real kappa_update = 0.0;
        for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = doc;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long) 25214903917 + 11;
            target = doc_table[(next_random >> 16) % table_size];
            if (target == doc) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) {
            f += syn0[c + word * layer1_size] * syn1doc[c + l2];
          }
          real tmp_kappa_update = f;
          f *= kappa[word];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          g *= global_lambda;
          kappa_update += g * tmp_kappa_update;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * kappa[word] * syn1doc[c + l2];
          for (c = 0; c < layer1_size; c++) syn1doc[c + l2] += g * kappa[word] * syn0[c + word * layer1_size];
        }
        for (c = 0; c < layer1_size; c++) syn0[c + word * layer1_size] += neu1e[c];
        if (with_kappa) {
          kappa[word] += kappa_update;
        }
      }
    }

    // regularization
    if (with_regularization && !is_pretrain) {
      if (word_counter == 0) {
        real scaling = -alpha * reg_lambda / (topics * num_per_topic);
        for (a = 0; a < topics; a++)
          for (b = 0; b < num_per_topic; b++) {
            int word_index = fix_seed_rankings[a * vocab_size + b] % vocab_size;
            for (c = 0; c < layer1_size; c++) {
              syn0[word_index * layer1_size + c] +=
                  scaling * delta_wik[(a * num_per_topic + b) * layer1_size + c];
            }
          }
        for (d = 0; d < topics; d++)
          for (c = 0; c < layer1_size; c++) {
            t_embeddings[d * layer1_size + c] += scaling * delta_tjk[d * layer1_size + c];
          }
      }
    }

    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  if (with_regularization && !is_pretrain) {
    free(wi);
    free(exp_rij);
    free(delta_tjk);
    free(delta_wik);
  }
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  long iter_count;
  FILE *fo, *fp;
  pthread_t *pt = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  printf("global: %d\n", with_global);
  if (with_global) {
    printf("Lambda for global is %lf\n", global_lambda);
  }
  printf("kappa: %d\n", with_kappa);
  if (with_kappa) {
    printf("Kappa is output to file %s\n", kappa_file);
  }
  printf("regularization: %d\n", with_regularization);
  if (with_regularization) {
    printf("Lambda for regularization is %lf\n", reg_lambda);
    printf("using topics in file %s\n", topic_file);
  }
  if (context_output[0] != 0) {
    printf("Output to: %s\n", context_output);
  }
  if (doc_output[0] != 0) {
    printf("Doc Output to: %s\n", doc_output);
  }
  if (output_tmp_file[0] != 0) {
    printf("output_tmp_file: %s\n", output_tmp_file);
  }

  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0){
    printf("No output file\n");
    return;
  }
  if (with_regularization && topic_output[0] == 0) {
    printf("Did not provide topic output file when topic is provided\n");
    return;
  }
  if (topic_file[0] != 0) {
    if (access(topic_file, R_OK) == -1) {
      printf("Topic file does not exist\n");
      return;
    }
  }
  // exit(0);
  InitNet();
  if (negative > 0) {
    InitUnigramTable();
    InitDocTable();
  }
  start = clock();
  if (with_regularization) {
    printf("Pretraining for %d epochs, in total %d + %lld = %lld epochs\n", pretrain_iters, pretrain_iters, iter,
           pretrain_iters + iter);
    iter += pretrain_iters;
  } else {
    pretrain_iters = iter;
  }
  is_pretrain = 1;
  if (pretrain_iters > 0) {
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *) a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  }

  if (with_regularization) {
    for (a = 0; a < topics; a++)
      for (b = 0; b < layer1_size; b++) {
        for (int i = topic_start_end[2 * a]; i < topic_start_end[2 * a + 1]; i++) {
          t_embeddings[a * layer1_size + b] += syn0[topic_index[i] * layer1_size + b];
        }
        t_embeddings[a * layer1_size + b] /= topic_start_end[2 * a + 1] - topic_start_end[2 * a];
      }
    printf("Training with regularization begins\n");
    is_pretrain = 0;
    FILE* fmid = fopen(output_tmp_file, "w");
    fclose(fmid);
    for (iter_count = pretrain_iters; iter_count < iter; iter_count++) {
      num_per_topic = (iter_count - pretrain_iters) * expand + initial_seed_set_size;
      similaritySearchSize = num_per_topic * 50;
      printf("Similarity Search size is %d\n", similaritySearchSize);
      for (a = 0; a < topics; a++)
        for (b = 0; b < vocab_size; b++) {
          wt_score[a * vocab_size + b] = 0;
          real norm = 0.0;
          for (c = 0; c < layer1_size; c++) {
            wt_score[a * vocab_size + b] += t_embeddings[a * layer1_size + c] * syn0[b * layer1_size + c];
            norm += syn0[b * layer1_size + c] * syn0[b * layer1_size + c];
          }
          wt_score[a * vocab_size + b] /= sqrt(norm);
        }
      if (rank_product && multi_keywords == 2) {
        printf("Rank product\n");
        for (a = 0; a < topics; a++) {
          topic_pivot_idx = topic_index[a]; // a = topic_start_end[a * 2]
          printf("topic index: %d\n", topic_pivot_idx);
          qsort(rankings1 + a * vocab_size, vocab_size, sizeof(int), kappaRankingComparator);
          for (b = 0; b < vocab_size; b++) kappa_rankings[rankings1[a * vocab_size + b]] = b + 1;
          qsort(rankings + a * vocab_size, vocab_size, sizeof(int), simRankingComparator);
          for (b = 0; b < vocab_size; b++) sim_rankings[rankings[a * vocab_size + b]] = b + 1;
          qsort(rankings + a * vocab_size, vocab_size, sizeof(int), productRankingComparator);
          printf("topic kappa: %f kapparank: %d\n", kappa[topic_pivot_idx], kappa_rankings[a*vocab_size+topic_pivot_idx]);
          // printf("beijing kappa: %f kapparank: %d\n", kappa[2441], kappa_rankings[a*vocab_size+2441]);
          // printf("top word in kapparank: %d, kappa: %f \n", rankings1[a*vocab_size] % vocab_size, kappa[rankings1[a*vocab_size] % vocab_size]);
          // printf("second word in rankproduct: %d, kappa: %f \n", rankings[a*vocab_size+1] % vocab_size, kappa[rankings[a*vocab_size+1] % vocab_size]);
        }
      } else {
        printf("Sim ranking\n");
        for (a = 0; a < topics; a++) qsort(rankings + a * vocab_size, vocab_size, sizeof(int), simRankingComparator);
      }
      if (fix_seed) {
        for (a = 0; a < topics; a++) {
          printf("Topic %d:\n", a);
          int words_in_topic = topic_start_end[a * 2 + 1] - topic_start_end[a * 2];
          int use_seed_topics = num_per_topic > words_in_topic ? words_in_topic : num_per_topic;
          printf("user input:\n");
          for (b = 0; b < use_seed_topics; b++) {
            fix_seed_rankings[a * vocab_size + b] = topic_index[topic_start_end[a * 2] + b];
            printf("%s, ",vocab[fix_seed_rankings[a * vocab_size + b]].word);
          }
          printf("\n");

          if (num_per_topic > words_in_topic) {
            printf("new found:\n");
            for (b = use_seed_topics, c = a * vocab_size; b < num_per_topic; b++) {
              while (1) {
                int not_used = 1;
                for (d = topic_start_end[0]; d < topic_start_end[2 * topics - 1]; d++) {
                  if (topic_index[d] == rankings[c] % vocab_size) {
                    not_used = 0;
                  }
                }
                if (not_used) break;
                c++;
              }
              fix_seed_rankings[a * vocab_size + b] = rankings[c] % vocab_size;
              c++;
              printf("%s, ", vocab[fix_seed_rankings[a * vocab_size + b]].word);
            }
            printf("\n");
          }
        }
      } else {
        for (a = 0; a < topics; a++) {
          for (b = 0; b < num_per_topic; b++) {
            fix_seed_rankings[a * vocab_size + b] = rankings[a * vocab_size + b] % vocab_size;
          }
          topic_pivot_idx = topic_index[a];
        }
      }
      FILE* fmid = fopen(output_tmp_file, "a");
      for (a = 0; a < topics; a++) {
        // printf("Cluster: %ld\n", a);
        for (b = 0; b < num_per_topic; b++) {
          fprintf(fmid, "%s ", vocab[fix_seed_rankings[a * vocab_size + b] % vocab_size].word);
          // printf("%s:%d,%f ", vocab[fix_seed_rankings[a * vocab_size + b] % vocab_size].word, kappa_rankings[rankings[a * vocab_size + b]], kappa[fix_seed_rankings[a * vocab_size + b] % vocab_size]);

        }
        // printf("top word in kapparank: %d, kappa: %f \n", rankings1[a*vocab_size] % vocab_size, kappa[rankings1[a*vocab_size] % vocab_size]);
        // printf("%d word in kapparank: %d, kappa: %f, product rank: %d  \n", kappa_rankings[rankings1[a*vocab_size] % vocab_size], rankings1[a*vocab_size] % vocab_size, kappa[rankings1[a*vocab_size] % vocab_size]);

        // printf("\n");
        fprintf(fmid, "\n");
      }
      fclose(fmid);
      for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *) a);
      for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    }
  }
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary)
        for (b = 0; b < layer1_size; b++) {
          fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
        }
      else
        for (b = 0; b < layer1_size; b++) {
          fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
        }
      fprintf(fo, "\n");
    }
    if (context_output[0] != 0) {
      FILE* fa = fopen(context_output, "wb");
      fprintf(fa, "%lld %lld\n", vocab_size, layer1_size);
      for (a = 0; a < vocab_size; a++) {
        fprintf(fa, "%s ", vocab[a].word);
        for (b = 0; b < layer1_size; b++) {
          fprintf(fa, "%lf ", syn1neg[a * layer1_size + b]);
        }
        fprintf(fa, "\n");
      }
      fclose(fa);
    }
    if (doc_output[0] != 0) {
      FILE* fa = fopen(doc_output, "wb");
      fprintf(fa, "%lld %lld\n", corpus_size, layer1_size);
      for (a = 0; a < corpus_size; a++) {
        fprintf(fa, "%d ", a);
        for (b = 0; b < layer1_size; b++) {
          fprintf(fa, "%lf ", syn1doc[a * layer1_size + b]);
        }
        fprintf(fa, "\n");
      }
      fclose(fa);
    }
    if (with_kappa) {
      fp = fopen(kappa_file, "wb");
      fprintf(fp, "%lld\n", vocab_size);
      for (a = 0; a < vocab_size; a++) {
        fprintf(fp, "%s ", vocab[a].word);
        fprintf(fp, "%lf\n", kappa[a]);
      }
      fclose(fp);
    }
    if (with_regularization) {
      fp = fopen(topic_output, "wb");
      fprintf(fp, "%d\n", topics);
      for (a = 0; a < topics; a++) {
        for (int i = topic_start_end[a * 2]; i < topic_start_end[a * 2 + 1]; i++) {
          fprintf(fp, "%s", vocab[topic_index[i]].word);
        }
        // fprintf(fp, "\t");
        for (b = 0; b < layer1_size; b++) {
          fprintf(fp, " %lf", t_embeddings[a * layer1_size + b]);
        }
        fprintf(fp, "\n");
      }
      fclose(fp);
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *) malloc(classes * sizeof(int));
    int *cl = (int *) calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *) calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return a;
    }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-kappa <file>\n");
    printf("\t\tUse <file> to save the resulting kappa's\n");
    printf("\t-topic <file>\n");
    printf("\t\tUse <file> to give the default topics, number of topics will be estimated from this\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-global_lambda <float>\n");
    printf("\t\tSet the relative importance of global context to local context; default is 0.5\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf(
        "./word2vec -train data.txt -output vec.txt -kappa kap.txt -topic topics.txt -size 200 -window 5 -sample 1e-4 -global_lambda 0.5 -negative 5 -binary 0-iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  output_tmp_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *) "-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-output", argc, argv)) > 0) {
    strcpy(output_file, argv[i + 1]);
    strcpy(output_tmp_file, argv[i + 1]);
    int filename_length = 0;
    for (int j=0; ;j++) {
      if(output_file[j] == 0) {
        filename_length = j;
        break;
      }
    }
    for (int j=filename_length; j>=0 ;j--) {
      if (output_file[j] != '.') {
        output_tmp_file[j+4] = output_file[j];
      } else {
        output_tmp_file[j+4] = output_file[j];
        output_tmp_file[j+3] = 'p';
        output_tmp_file[j+2] = 'm';
        output_tmp_file[j+1] = 't';
        output_tmp_file[j] = '_';
        break;
      }
    }
  }

  if ((i = ArgPos((char *) "-global_lambda", argc, argv)) > 0) global_lambda = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-reg_lambda", argc, argv)) > 0) reg_lambda = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-kappa", argc, argv)) > 0) {
    strcpy(kappa_file, argv[i + 1]);
    with_kappa = 1;
  }
  if ((i = ArgPos((char *) "-topic", argc, argv)) > 0) {
    strcpy(topic_file, argv[i + 1]);
    with_regularization = 1;
  }
  if ((i = ArgPos((char *) "-topic_output", argc, argv)) > 0) {
    strcpy(topic_output, argv[i + 1]);
  }
  if ((i = ArgPos((char *) "-pretrain", argc, argv)) > 0) pretrain_iters = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-rank_product", argc, argv)) > 0) rank_product = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-gen_vocab", argc, argv)) > 0) gen_vocab = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-load_emb", argc, argv)) > 0) load_emb = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-load_emb_with_v", argc, argv)) > 0) load_emb_with_v = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-load_emb_file", argc, argv)) > 0) strcpy(load_emb_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-fix_seed", argc, argv)) > 0) fix_seed = atoi(argv[i + 1]);


  if ((i = ArgPos((char *) "-context", argc, argv)) > 0) strcpy(context_output, argv[i + 1]);
  if ((i = ArgPos((char *) "-doc_output", argc, argv)) > 0) strcpy(doc_output, argv[i + 1]);

  if ((i = ArgPos((char *) "-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int));
  docs = (int *) calloc(corpus_max_size, sizeof(int));
  doc_sizes = (long long *) calloc(corpus_max_size, sizeof(long long));
  expTable = (real *) malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
