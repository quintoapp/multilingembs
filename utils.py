from nltk.tokenize import word_tokenize
import os
import io
import numpy as np

def load_vec(emb_path, nmax=50000):
	vectors = []
	word2id = {}
	with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
		next(f)
		for i, line in enumerate(f):
			word, vect = line.rstrip().split(' ', 1)
			vect = np.fromstring(vect, sep=' ')
			assert word not in word2id, 'word found twice'
			vectors.append(vect)
			word2id[word] = len(word2id)
			if len(word2id) == nmax:
				break
	id2word = {v: k for k, v in word2id.items()}
	embeddings = np.vstack(vectors)
	return embeddings, id2word, word2id

def tokenize(sentence):
	return word_tokenize(sentence)

def fetch_vectors(lang, sentence):
	tokens = tokenize(sentence)
	emb, id2word, word2id = load_vec(os.path.join('data', 'vectors-' + lang + '.txt'))
	emb_vec = [emb[word2id.get(i, word2id['unknown'])] for i in tokens]
	return emb_vec

def fetch_vectors_v1(lang, sentence):
	tokens = tokenize(sentence)
	emb, id2word, word2id = load_vec(os.path.join('data', 'vectors-' + lang + '.txt'))
	emb_vec = [emb[word2id.get(i, 'unknown')] for i in tokens]
	return np.mean(emb_vec, axis=0)

def similarity(v1, v2):
	return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

"""
def evaluate(file_path):
	import pandas
	eval_f = pandas.read_excel(file_path)

	pass
"""
