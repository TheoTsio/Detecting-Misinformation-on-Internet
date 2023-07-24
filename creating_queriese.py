from partB import *

queries_title = []
queries_description = []
queries_narrative = []
with open("misinfo-2021-topics.xml", 'r') as file:
    for line in file:
        if line.startswith('<query>'):
            queries_title.append(line)
        if line.startswith('<description>'):
            queries_description.append(line)
        if line.startswith('<narrative>'):
            queries_narrative.append(line)
    


with open('trec_topics_terrier_with_lesk_algorithm.txt', 'w') as file:
    for i in range(len(queries_title)):
        file.write('<top>\n')
        file.write('<num>' + str(i + 1 + 100) + '</num>\n')
        file.write('<title>' + query_extension(queries_title[i][len('<query>'): -len('<\query>\n')]) + '</title>\n')
        file.write('<desc>' + query_extension(queries_description[i][len('<description>'): -len('</description>\n')]) + '</desc>\n')
        file.write('<narr>' + query_extension(queries_narrative[i][len('<narrative>'): -len('</narrative>\n')]) + '</narr>\n')
        file.write('</top>\n')

