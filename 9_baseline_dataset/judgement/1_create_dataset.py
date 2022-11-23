import re
import numpy as np


""" 
This dataset is special, because it contains wrong data!
The authors originally used it to identify this data,
so we'll have to filter it first. How? Well, luckily 
this is specified in their original publication.
https://aclanthology.org/J14-2003.pdf
https://users.monash.edu/~ingrid/Publications/SeroussiSmythZukerman.pdf

Keep all entries from "Dixon"
Keep all McTiernan fomr 1965 to 1975
Keep all Rich from 1913 to 1928
"""
valid = ["McTiernan1965-1975", "Dixon", "Rich1913-1928"]


def writeToFile(authors, features, author2id, paper_ids, content):
    """ Writes all data into files
    The following files are written
    - dataset/features.csv
    - dataset/authors.txt
    - dataset/paper_ids.csv
    - dataset/author2id.csv
    - dataset/content.csv
    For file specification, see README.md in root directory.
    """
    np.savetxt("dataset/" + "features.csv", features, fmt="%1i")
    np.savetxt("dataset/" + "authors.txt", authors, fmt="%s")
    np.savetxt("dataset/" + "paper_ids.csv", paper_ids, fmt="%s")
    np.savetxt("dataset/" + "author2id.csv",
            np.array(author2id[authors[0]], dtype=str, ndmin=2), fmt="%s", newline='')
    with open("dataset/" + "author2id.csv", "ab") as f:
        f.write(b"\n")
        for i in range(1, len(authors)):
            np.savetxt(f, np.array(author2id[authors[i]], dtype=str, ndmin=2), fmt="%s")
    with open("dataset/"+ "content.csv", "w") as f:
        f.write("\n".join(content))
        f.write("\n")




with open("judgement.txt") as f:
    data = [x.split("\t") \
            for x in f.read().strip("\n").splitlines() \
            if any([v in x[:100] for v in valid])]

authors = ["McTiernan", "Dixon", "Rich"]
author2id = {key: [] for key in authors}
content = []
paper_ids = []
feature_vec = np.zeros((len(data), len(authors)))

for i,text in enumerate(data):
    author = authors[[a in text[0] for a in authors].index(1)]
    paper_id = text[1].strip()
    text = text[2].strip()
    
    author2id[author].append(paper_id)
    paper_ids.append(paper_id)
    feature_vec[i, authors.index(author)] = 1
    content.append(text)

writeToFile(authors, feature_vec, author2id, paper_ids, content)


