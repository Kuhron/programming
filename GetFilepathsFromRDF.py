import os
import re

# just print the filepaths in this script, relative to the repository on EHD

ehd_repository_dir = "D:\\LearningForPNG\\repository"
rdf_repository_dir = "/home/wesley/Desktop/Learning/repository/"
pattern = "<rdf:resource rdf:resource=\"{rdf_repository_dir}.*\"/>"

with open("/home/kuhron/horokoi/Zotero/HorokoiLVCs.rdf") as f:
    lines = f.readlines()

matches = []
for line in lines:
    m = re.search(pattern, line)
    matches.append(m)
print(matches)
