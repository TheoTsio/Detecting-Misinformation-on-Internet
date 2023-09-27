from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from lesk_py_files.lesk_algorithm_unit import *
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

def lesk_algorithm(sentence):
	stop_words = set(stopwords.words('english'))
	query = word_tokenize(sentence)
	list_of_words_to_be_extended = []
	for word in query:
		if word not in stop_words:

			apply_lesk = lesk_algorithm_unit(query, word, 'n')
			if apply_lesk:
				list_of_words_to_be_extended += [lemma.name() for lemma in apply_lesk.lemmas()]
			else:
				list_of_words_to_be_extended += []
	return list_of_words_to_be_extended
	
def query_extension(initial_query):
	list_of_words_to_be_extended = lesk_algorithm(initial_query)
	for word in list_of_words_to_be_extended:
		if word not in initial_query:
			initial_query += ' '
			initial_query += word
	return initial_query


if __name__ == '__main__':
	print(query_extension('vitamin b12 sun exposure vitiligo'))



