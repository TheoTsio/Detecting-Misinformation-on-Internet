import gzip
import csv
import glob

#I am loading the initial .csv file(the file without the document text) 
qrels = []
f = open("qrels_data_sort_no_dupl.csv", "r")
for count, inpu in enumerate(f):
	if count % 2 == 0:
		qrels.append(inpu.split(",")[2])

path = "/var/scratch3/corpora/trec21_health_misinfo/c4"
pattern = "?????"
files = sorted(list(glob.iglob(f'{path}/en.noclean.trec.format/c4-train.{pattern}-of-07168.json.gz')))

#I am creating a hash table with keys the number of gzip and values the documents from this gzip that i want

hash_table_gz = {}
help_list = []
current_gz = "00000"
for i in qrels:
	if i[9:14] == current_gz:
		help_list.append(int(i[24:].split(" ")[0]))
	else:	
		help_list.sort()
		hash_table_gz[current_gz] = help_list
		help_list = []
		help_list.append(int(i[24:].split(" ")[0]))
		current_gz = i[9:14]


#I am parsing the gzip files
temp = []
res = []

for count1, filepath in enumerate(files):
	with gzip.open(filepath) as o:
		for count, line in enumerate(o):
			line = line.decode('utf-8')
			try:
				if hash_table_gz[filepath[79:84]] == []:
					break
				elif count == hash_table_gz[filepath[79:84]][0]:
					hash_table_gz[filepath[79:84]].pop(0)
					print(qrels[len(res)])					
					temp.append(qrels[len(res)])
					temp.append(line)
					res.append(temp)
					temp =[]						
			except:
				continue
f = open("qrels_data_test_final.csv", "w", newline="") # I am creating a CSV file.
writer = csv.writer(f)

for i in res:
    writer.writerow(i)

f.close()
					
