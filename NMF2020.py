import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import re
import string
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


df = pd.read_csv('corpus.csv')
df = df.rename(columns={df.columns[0]: 'Name'})
df = df[['Name', 'transcripts']]
df = df.set_index('Name')


def clean_text_round1(text):
	text = text.lower()
	text = re.sub('\[.*?\]', ' ', text)
	text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
	text = re.sub('\w*\d\w*', ' ', text)
	return text


df["transcripts"] = df["transcripts"].apply(lambda x: clean_text_round1(x))


def nouns(text):
	is_noun = lambda pos:pos[:2]=='NN'
	tokenized = word_tokenize(text)
	wordnet_lemmatizer = WordNetLemmatizer()
	all_nouns = [wordnet_lemmatizer.lemmatize(word) for (word, pos) in pos_tag(tokenized) if is_noun(pos)]
	return ' '.join(all_nouns)


data_nouns = pd.DataFrame(df.transcripts.apply(nouns))

tv_noun = TfidfVectorizer(ngram_range = (1,1), max_df = .8, min_df = .01)
data_tv_noun = tv_noun.fit_transform(data_nouns.text)
data_dtm_noun = pd.DataFrame(data_tv_noun.toarray(), columns=tv_noun.get_feature_names())
data_dtm_noun.index = df.index


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


n_rank = 5

nmf_model = NMF(n_rank)
doc_topic = nmf_model.fit_transform(data_dtm_noun)
# display_topics(nmf_model, tv_noun, 5)
display_topics(nmf_model, tv_noun.get_feature_names(), 5)
