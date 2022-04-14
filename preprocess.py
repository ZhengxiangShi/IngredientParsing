import tqdm
import nltk
import torch
import random
import numpy as np
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset


def get_augmented_samples(file, size=10000):
    label2id = {'DF': 0, 'NAME': 1, 'O': 2, 'QUANTITY': 3, 'SIZE': 4, 'STATE': 5, 'TEMP': 6, 'UNIT': 7}
    id2label = {v: k for k, v in label2id.items()}
    distribution = {}
    distribution_of_length = {}
    for i in range(10):
        distribution[i] = {}
        for key in label2id.keys():
            distribution[i][key] = 0
    candidate_pool = defaultdict(set)
    count = 0
    
    with open(file, 'r') as f:
        lines = f.readlines() 
        for line in lines:
            line = line.strip().strip('\n')
            if line:
                token, tag = line.split('\t')
                token = convert_number(token)
                token = token.lower()
                candidate_pool[tag].add(token)
                if count < 10:
                    distribution[count][tag] += 1 
                count += 1
            else:
                if count != 0 and count < 10:
                    if count in distribution_of_length:
                        distribution_of_length[count] += 1
                    else:
                        distribution_of_length[count] = 1
                count = 0                
    
    samples = []
    for k in tqdm.tqdm(range(size)):
        label = []
        sentence = []
        phrase_length = \
            random.choices([length for length in distribution_of_length.keys()],
                           weights=[prob for prob in distribution_of_length.values()])[0]
        for j in range(phrase_length):
            index = random.choices(range(8), weights=[prob for prob in distribution[j].values()])[0]
            selected_label = id2label[index]
            selected_word = random.choices(list(candidate_pool[selected_label]))[0]
            sentence.append(selected_word)
            label.append(index)
        sample = {'text': " ".join(sentence), 'labels': label}
        samples.append(sample) 
                    
    for sample in samples:
        sample['text'] = sample['text'].replace(' ,', ',')
        
    return samples


def get_samples(file):
    label2id = {'DF': 0, 'NAME': 1, 'O': 2, 'QUANTITY': 3, 'SIZE': 4, 'STATE': 5, 'TEMP': 6, 'UNIT': 7}
    samples = []
    label_types = set()
    
    with open(file, 'r') as f:
        lines = f.readlines()
        label = []
        sentence = []
        for line in lines:
            line = line.strip().strip('\n')
            if not line:
                if label and sentence:
                    sample = {'text': " ".join(sentence), 'labels': label}
                    samples.append(sample)
                label = []
                sentence = []
            else:
                token, tag = line.split('\t')
                token = convert_number(token)
                token = token.lower()
                if len(token.split()) > 1:
                    tokensplit = token.split()
                    for tokensplit_item in tokensplit:
                        sentence.append(tokensplit_item)
                        label.append(label2id[tag])
                else:
                    sentence.append(token)
                    label.append(label2id[tag])
                    label_types.add(tag)
        sample = {'text': " ".join(sentence), 'labels': label}
        samples.append(sample)           
                    
    for sample in samples:
        sample['text'] = sample['text'].replace(' ,', ',')
        
    return samples, label_types


class Vocabulary(object):
    def __init__(self, data=None, vector_filename='./data/glove.42B.300d.txt', embed_size=300):
        self.vector_filename = vector_filename
        self.embed_size = embed_size
        self.word2idx = {}
        self.idx2word = {}

        self.word_counts = defaultdict(int)
        self.tokenized_data = []
        self.oov_words = set()

        self.get_dataset_properties(data) 
        self.init_vectors() 
        self.load_vectors() 
        self.add_oov_vectors()

        self.word_embeddings = nn.Embedding(self.word_vectors.shape[0], self.word_vectors.shape[1])
        self.word_embeddings.weight.data.copy_(torch.from_numpy(self.word_vectors))
        self.num_tokens = len(self.word2idx)

    def get_dataset_properties(self, data):
        for sentence in data:
            tokenized_sentence = nltk.word_tokenize(sentence)
            for token in tokenized_sentence:
                self.tokenized_data.append(token)
                self.word_counts[token] += 1

    def init_vectors(self):
        embed_size = self.embed_size
        vector_arr = np.zeros((1, embed_size))
        self.add_word('<pad>') 
        self.word_vectors = vector_arr

    def load_vectors(self):
        vector_filename = self.vector_filename
        embed_size = self.embed_size
        f = open(vector_filename, 'r')

        data = []
        data_rare = []
        for line in f:
            tokens = line.split()
            if len(tokens) != embed_size+1:
                continue
            word = tokens[0]
            if word not in self.word_counts:
                if word in self.word_counts:
                    for t in tokens[1:]:
                        data_rare.append(float(t))
                continue
            for t in tokens[1:]:
                data.append(float(t))
            self.add_word(word)

        data_arr = np.reshape(data, newshape=(int(len(data)/embed_size), embed_size)) 
        self.word_vectors = np.concatenate((self.word_vectors, data_arr), 0)
        if data_rare:
            data_rare_arr = np.reshape(data_rare, newshape=(int(len(data_rare)/embed_size), embed_size))
            avg_vector = np.expand_dims(np.mean(data_rare_arr, axis=0), axis=0)
            self.add_unk(avg_vector)
        else:
            print("Adding random vector for rare bucket as there are no rare words...")
            self.add_unk(np.random.rand(1, embed_size))

    def add_unk(self, avg_vector):
        self.word_vectors = np.concatenate((self.word_vectors, avg_vector), 0)
        self.add_word('<unk>')

    def add_oov_vectors(self):
        embed_size = self.embed_size
        for word in self.tokenized_data:
            if word not in self.word2idx:
                self.word_vectors = np.concatenate((self.word_vectors, np.random.rand(1, embed_size)), 0)
                self.add_word(word)
                self.oov_words.add(word)

    def add_word(self, word):
        idx = len(self.word2idx)
        self.word2idx[word] = idx
        self.idx2word[idx] = word

    def __call__(self, sentence):
        input_ids = []
        tokenized_sentence = nltk.word_tokenize(sentence)
        for token in tokenized_sentence:            
            if not token in self.word2idx:
                input_ids.append(self.word2idx['<unk>'])
            else:
                input_ids.append(self.word2idx[token])
        return input_ids
    
    def decode(self, ids):
        sentence = []
        for id_ in ids:
            sentence.append(self.idx2word[id_])
        return sentence

    def __len__(self):
        return len(self.word2idx)

    def __str__(self):
        vocabulary = ""
        for key in self.idx2word.keys():
            vocabulary += str(key) + ": " + self.idx2word[key] + "\n"
        return vocabulary

   
class IngredientDataset(Dataset):
    def __init__(self, data, vocab):
        self.samples = []
        tag2ids = {'RBS': 0, 'CD': 1, 'SYM': 2, 'DT': 3, '$': 4, "''": 5, 'TO': 6, 'PDT': 7, 'WP': 8, 'RBR': 9,
                   'POS': 10,
                   'VBD': 11, 'PRP$': 12, 'IN': 13, 'VBN': 14, 'VBP': 15, 'FW': 16, 'JJR': 17, 'NNP': 18, 'JJS': 19,
                   'VBZ': 20, 'RP': 21, 'WRB': 22, 'LS': 23, 'RB': 24, '.': 25, 'NN': 26, 'PRP': 27, 'JJ': 28, 'VB': 29,
                   'CC': 30, 'MD': 31, ',': 32, '``': 33, 'WDT': 34, 'VBG': 35, ':': 36, 'NNS': 37, 'NNPS': 38}

        for sample in data:
            input_ids = torch.tensor(vocab(sample['text']))
            sent = preprocess(sample['text'])
            assert len(sent) == len(input_ids)
            tag_ids = torch.tensor(list(map(lambda x: tag2ids.get(x[1], 0), sent)))
            label = torch.tensor(sample['labels'])
            try:
                assert len(input_ids) == len(label)
            except:
                continue
            self.samples.append((input_ids, tag_ids, label))
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx][0], self.samples[idx][1], self.samples[idx][2]


def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent


def convert_number(ingredient):
    ingredient = ingredient.replace("½", "0.5")
    ingredient = ingredient.replace("⅓", "0.33")
    ingredient = ingredient.replace("⅔", "0.67")
    ingredient = ingredient.replace("¼", "0.25")
    ingredient = ingredient.replace("¾", "0.75")
    ingredient = ingredient.replace("⅕", "0.4")
    ingredient = ingredient.replace("⅖", "0.4")
    ingredient = ingredient.replace("⅗", "0.6")
    ingredient = ingredient.replace("⅘", "0.8")
    ingredient = ingredient.replace("⅞", "0.875")
    return ingredient
