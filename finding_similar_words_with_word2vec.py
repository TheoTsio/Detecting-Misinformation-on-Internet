from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def expand_query_with_similar_words_with_word2vec(query):
    # Load pre-trained Word2Vec model
    model = Word2Vec.load("healthword2vec_cbow.model")

    expanded_query = ''
    # Preprocessed query words
    query_words = query.split()
    # query_words = ['serepiditly']

    terms_to_expand = []

    # Iterate through each query word
    for query_word in query_words:
        if query_word in model.wv.key_to_index: # one of the most important limitations of Word2Vec is the inability to generate vectors for words not present in the vocabulary (called OOV — out of vocabulary words).
            # query_word_embedding = model.wv[query_word].reshape(1, -1)

            similar_words = model.wv.most_similar(positive=[query_word], topn=1)
            # Calculate cosine similarity between query word and all other words
            # similarities = cosine_similarity(query_word_embedding, model.wv.vectors)

            # # Find N most similar words
            # N = 3  # Number of similar words to consider
            # similar_word_indices = np.argsort(similarities[0])[::-1][:N]
            # similar_words = [model.wv.index_to_key[idx] for idx in similar_word_indices]

            terms_to_expand += [word[0] for word in similar_words]
            terms_to_expand = list(set(terms_to_expand)) # Because set has no duplicates and i want to remove duplicates


    for term in terms_to_expand:
        if term[:-1] not in query: # i am only expanding the terms that do not exist already in the query
            expanded_query += ' ' + term
    
    return expanded_query

if __name__ == '__main__':
    query = 'vitamin b12 sun exposure vitiligo'
    print(expand_query_with_similar_words_with_word2vec(query))