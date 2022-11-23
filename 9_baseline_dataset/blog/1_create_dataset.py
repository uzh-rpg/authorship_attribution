import os
import re
import numpy as np
import pandas as pd
from sortedcontainers import SortedList

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

data = pd.read_csv("blogtext.csv")
data['text'] = [x.strip() for x in data['text']]

authors = list(set([author for author in data['id']]))
author2id = {author: [] for author in authors}

ids = data['id']
texts = data['text']
authors = SortedList(authors)
for i in range(len(data)):
    author = ids[i]
    author2id[author].append(i)

length_df = pd.DataFrame()
length_df['author'] = authors
length_df['length'] = [len(author2id[x]) for x in authors]
length_df = length_df.sort_values(by=['length'], ascending=False).iloc[0:50]

authors = list(length_df['author'])
author2id = {author: [] for author in authors}
feature_vec = np.zeros((len(data), len(authors)))
content = []
paper_ids = []
for i in range(len(data)):
    author = ids[i]
    text = texts[i]
    if author not in authors:
        continue
    if re.search(r"^\s*$", text):
        continue
    author2id[author].append(i)
    paper_ids.append(i)
    feature_vec[i, authors.index(author)] = 1
    content.append(texts[i])

feature_vec = feature_vec[np.sum(feature_vec, axis=1) == 1,:]
writeToFile(authors, feature_vec, author2id, paper_ids, content)
