def lesk_algorithm(sentence):
    query = word_tokenize(sentence)
    apply_lesk = lesk_algorithm_unit(query, "vitiligo")
    if apply_lesk:
        print([lemma.name() for lemma in apply_lesk.lemmas()])
    else:
        print([])