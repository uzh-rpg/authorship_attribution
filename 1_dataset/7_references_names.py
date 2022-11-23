import os
import re
import random
import numpy as np
from collections import Counter
from itertools import compress

with open("dataset/references.csv") as f:
    references_all = [x.split("####") for x in f.read().strip("\n").split("\n")]

""" Ways to end the name list
replace and with ,
replace ,, with ,
replace ^\s*\[.*?\]\s* with
replace ^\s*\(.*?\)\s* with


"
et al
et al.
et. al.
>3 capital letters
\d
\s[A-Z]\. und \s[A-Z]\..\[A-Z]\.

Possible author sep chars
, & ;
"""

references_new = []
for refs in references_all:
    refs = [re.sub(r" and", ", ", ref) for ref in refs]
    refs = [re.sub(r",\s*,", ", ", ref) for ref in refs]
    refs = [re.sub(r"^\s*\[.*?\]\s*", "", ref) for ref in refs]
    refs = [re.sub(r"^\s*\(.*?\)\s*", "", ref) for ref in refs]
    refs = [re.sub(r"\".*$", "", ref) for ref in refs]
    refs = [re.sub(r"http.*$", "", ref) for ref in refs]
    refs = [re.sub(r"et\.?\s+al.*$", "", ref) for ref in refs]
    refs = [re.sub(r"[A-Z]{3,}.*$", "", ref) for ref in refs]
    refs = [re.sub(r"\d.*$", "", ref) for ref in refs]
    refs = [re.sub(r"\s[A-Z]\..?[a-zA-Z]\.", " ", ref) for ref in refs]
    refs = [re.sub(r"^[A-Z]\..?[a-zA-Z]\.", " ", ref) for ref in refs]
    refs = [re.sub(r"\s[A-Z]-[a-zA-Z]\.?", " ", ref) for ref in refs]
    refs = [re.sub(r"^[A-Z]-[a-zA-Z]\.?", " ", ref) for ref in refs]
    refs = [re.sub(r"\s[A-Z]\.", " ", ref) for ref in refs]
    refs = [re.sub(r"^[A-Z]\.", " ", ref) for ref in refs]
    refs = [re.sub(r"\s[A-Z]\s", " ", ref) for ref in refs]
    refs = [re.sub(r"^[A-Z]\s", " ", ref) for ref in refs]
    refs = [re.sub(r"\s+" ," ", ref) for ref in refs]
    refs = [re.sub(r"^\s+" ,"", ref) for ref in refs]

    joined = " ".join(refs)
    seps = [",", "&", ";", "."]
    counter = [joined.count(sep) for sep in seps]
    sep = seps[counter.index(max(counter))]

    for i in range(len(refs)):
        names = refs[i].split(sep)
        mask = [name.strip().count(" ") < 2
                and not "." in name
                and re.search(r"[A-Z][a-z]", name)
                and not re.search(r"\b[a-z]", name) for name in names]
        names = list(compress(names, mask))
        names = [re.sub(r"\b\w[\b']", "", name) for name in names]
        names = [re.sub(r"\b[A-Z]{2}\b", "", name) for name in names]
        names = [re.sub(r"[^\w\s\-']", "", name).strip() for name in names]
        refs[i] = "|".join(names)
    refs = [ref for ref in refs if ref != ""]
    references_new.append(refs)

string_list = ["####".join(x) for x in references_new]
string = "\n".join(string_list) + "\n"
string = string.lower()

with open("dataset/reference_authors.csv", "w") as f:
    f.write(string)


# Now find the null-authors
with open("dataset/reference_authors.csv") as f:
    index = [i for i,x in enumerate(f.read().strip("\n").splitlines()) if x.strip() == ""]

with open("dataset/reference_null.csv", "w") as f:
    f.write("\n".join([str(i) for i in index]) + "\n")


with open("dataset/reference_authors.csv") as f:
    reference_authors = f.read().strip("\n")

# Prepare the vocab
data = reference_authors.replace("\n","|").replace("####","|")
vocab = list(set(data.split("|")))
random.shuffle(vocab)
vocab_dict = {word: idx for idx,word in enumerate(vocab)}

reference_authors = reference_authors.splitlines()
embeddings = np.zeros((len(reference_authors), len(vocab_dict.keys())), dtype=int)

for i,paper in enumerate(reference_authors):
    authors = paper.replace("####","|").split("|")
    indices = np.array([vocab_dict[author] for author in authors])
    unique, counts = np.unique(indices, return_counts=True)
    for index,n in zip(unique, counts):
        embeddings[i,index] = n


tmp = embeddings.sum(axis=0)
ind = np.argpartition(tmp, -20)[-20:]
ind = ind[np.argsort(tmp[ind])]
[print(tmp[i], vocab[i]) for i in ind[::-1]]


embeddings = embeddings[:, embeddings.sum(axis=0) > 50]

print(embeddings.shape)
np.savetxt("dataset/reference_count_embeddings.csv", embeddings, fmt="%i")
