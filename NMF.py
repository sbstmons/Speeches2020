import numpy as np
import pandas as pd
from sklearn.decomposition import NMF


def update_H(W, H, V):
	numerator = W.T.dot(V)
	denominator = W.T.dot(W).dot(H) + 1e-10
	if isinstance(numerator, pd.DataFrame):
		numerator.columns = [*range(numerator.shape[1])]
	H = H*(numerator / denominator)
	return H


def update_W(W, H, V):
	V.columns = [*range(V.shape[1])]
	numerator = V.dot(H.T)
	denominator = W.dot(H).dot(H.T) + 1e-10
	W = W*(numerator / denominator)
	return W


def do_nnmf(V, rank=10, iter=100):
	n, m = V.shape

	W = np.abs(np.random.randn(1, n, rank))[0]
	H = np.abs(np.random.randn(1, rank, m))[0]

	for i in range(iter):
		H = update_H(W, H, V)
		W = update_W(W, H, V)

	return H, W


def display_topics(model, feature_names, num_top_words, topic_names=None):
	for ix, topic in enumerate(model.components_):
		#print topic, topic number, and top words
		if not topic_names or not topic_names[ix]:
			print("\nTopic ", ix)
		else:
			print("\nTopic: '",topic_names[ix],"'")
		print(", ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))


def display_topics_New(H, feature_names, num_top_words):
	for ix, topic in enumerate(H):
		print("\nTopic ", ix)
		print(", ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))


data_dtm_noun = pd.read_csv('nounsCorpus.csv')
tv_noun = open("allNouns.txt", "r").read().split("\n")
tv_noun.remove("today")
data_dtm_noun = data_dtm_noun.set_index('Name')

n_rank = 5

nmf_model = NMF(n_rank)
doc_topic = nmf_model.fit_transform(data_dtm_noun)
display_topics(nmf_model, tv_noun, 5)

print("\n ****** ")

H, W = do_nnmf(data_dtm_noun, rank=n_rank, iter=300)
V_rec = W.dot(H)
#V_rec.columns = tv_noun.remove("today")
display_topics_New(H.to_numpy(), tv_noun, 5)
