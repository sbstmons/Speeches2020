import os
import csv

your_path = 'dem2020'
files = os.listdir('dem2020')
fields = ['Name', 'Text']
rows = []

for file in files:
	if os.path.isfile(os.path.join(your_path, file)):
		f = open(os.path.join(your_path, file), 'r')
		s = f.read()
		rows.append([file[:-4], s])
		f.close()

with open("dem2020.csv", 'w') as csvfile:
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(fields)
	csvwriter.writerows(rows)
