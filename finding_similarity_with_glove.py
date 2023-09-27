import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    with open(gloveFile, encoding='utf8') as f:
        content = f.readlines()
    model = {}
    for line in content:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        model[word] = embedding
    print('Done.', len(model), ' words loaded!')
    return model

def expand_query_with_similar_words_with_glove(query, model, top_n=3):
    words = query.split(" ")
    similar_words = []

    for target_word in words:
        print('Finding similar words for', target_word)
        try:
            target_embedding = model[target_word]
        except:
            print(target_word, "not in vocabulary")
            continue
        
        # Batch processing for cosine similarity
        embeddings = np.array(list(model.values()))
        similarities = cosine_similarity([target_embedding], embeddings)[0]
        similar_word_indices = similarities.argsort()[::-1][:top_n]

        for idx in similar_word_indices:
            similar_words.append(list(model.keys())[idx])

    for similar_word in similar_words:
        if similar_word[:-2] not in query and similar_word is not None:
            query += " " + similar_word
    return query


if __name__ == '__main__':
    query = 'starve a fever, feed a cold'
    model = loadGloveModel('glove.42B.300d.txt')
    print(expand_query_with_similar_words_with_glove(query, model, 3))
