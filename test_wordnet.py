from nltk.corpus import wordnet as wn

sets = wn.synsets("wind")

for synset in sets:
    synonyms = []
    for lemma in synset.lemmas():
        synonyms.append(lemma.name())
    print(synonyms)