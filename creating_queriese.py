from lesk_py_files.lesk_algorithm_iplementation import *
# from word2vec_py_files.finding_similar_words_with_word2vec import *
# from finding_similarity_with_glove import *

queries_title = []
queries_description = []
queries_narrative = []
with open("misinfo-2021-topics.xml", 'r') as file:
    print("hello")
    for line in file:
        if line.startswith('<query>'):
            queries_title.append(line)
        if line.startswith('<description>'):
            queries_description.append(line)
        if line.startswith('<narrative>'): #and line.find('A very useful document'):
            queries_narrative.append(line) #[:line.find('A very useful document')])


# model = loadGloveModel('glove.42B.300d.txt')
with open('query_extension_test.txt', 'w') as file:
    for i in range(len(queries_title)):
        print("Query", str(i + 1 + 100))
        file.write('<top>\n')
        file.write('<num>' + str(i + 1 + 100) + '</num>\n')
        file.write('<title>' + query_extension(queries_title[i][len('<query>'): -len('<\query>\n')]) + '</title>\n')
        file.write('<desc>' + queries_description[i][len('<description>'): -len('</description>\n')] + '</desc>\n')
        
        file.write('<narr>' + queries_narrative[i][len('<narrative>'): -len('</narrative>\n')] + '</narr>\n') #i will use this line for the original query creation but in the modified narrative i remove everything from the "A very usefull document and so i dont need to change it here" 
        # file.write('<narr>' + queries_narrative[i][len('<narrative>'):-1] + '</narr>\n') 
        file.write('</top>\n')


