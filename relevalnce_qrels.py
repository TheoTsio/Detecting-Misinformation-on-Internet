with open("misinfo-qrels-graded.usefulness", 'r') as file:
    data = file.readlines()
print(data)
new_data = []
c1 = 0
for line in data:
    new_line = line.split(" ")
    new_line[2] = new_line[2][20:25] + new_line[2][35:]
    print(" ".join(new_line))
    new_data.append(" ".join(new_line))
    c1 += 1

with open("relevance_qrels.txt", "w") as file:
    for line in new_data:
        file.write(line)

