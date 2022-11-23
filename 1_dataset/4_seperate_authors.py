""" Summary
Because the arXiv dataset does not contain unique author identifiers, the dataset
inherently suffers from the problem "same name, different person". To overcome 
this, the following approach is used:
1. For all abstracts of papers associated with the same name (i.e. all papers
   listed in the same line of the author2id.csv file, use a pretrained 
   transformer from Huggingface ('all-mpnet-base-v2') to calculate feature 
   embeddings.
2. Using the DBSCAN algorithm, the embeddings for a single name are clustered.
3. If there are too many clusters or too many unclustered points, the author
   is discarded. This approach removes about 25-30% of the dataset, but mostly
   Asian names as a name like "Li Fei" is very ambiguous compared to "Pieter Abeel"
   or "Yann LeCun".

Note on the clustering: the clustering algorithm and the threshold parameters for 
the 3rd step are hand tuned. To do so, a list of known "non ambigious authors" and
a list of ambiguous authors (checked via google scholar) was used. Then, the algorithm
was hand-tuned to yield a good performance.
"""

import numpy as np
import pickle as pkl
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sentence_transformers import SentenceTransformer, util

""" Read the abstracts and compute the feature embeddings. 
The embeddings are saved to a pickle-file. The transformer
used was chosen based on a benchmark comparison published
on huggingface: 
     https://www.sbert.net/_static/html/models_en_sentence_embeddings.html
It achieved the best performance on sentence embedding.
"""
def embed():
    with open("dataset/paper_ids.csv") as f:
        paper_ids = f.read().strip("\n").split("\n")

    with open("dataset/abstracts.csv") as f:
        abstracts = f.read().strip("\n").split("\n")

    tmp={}
    for i,a in zip(paper_ids, abstracts):
        tmp[i] = a
    abstracts = tmp

    with open("dataset/author2id.csv") as f:
        a2i = f.read().strip("\n").split("\n")

    with open("dataset/authors.txt") as f:
        authors = f.read().strip("\n").split("\n")

    data = {}
    for author, paper_list in zip(authors, a2i):
        papers = paper_list.strip().split(" ")
        data[author] = []
        for paper in papers:
            data[author].append(abstracts[paper])

    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = []
    for idx in range(0, len(authors)):
        texts = data[list(data.keys())[idx]]
        print(len(texts))
        embeddings.append(
                model.encode(texts, convert_to_tensor=True
            ).cpu().numpy())

    pkl.dump(embeddings, open("embeddings.pkl","wb"))


""" Tuning Function
This function is not called during a standard run of the script. It can be used
to tune the parameters of the clustering algorithm and the threshold parameters
for author rejection.
For this, manually provide the indices for authors that are known to be ambigious
("bad_authors") and non-ambiguous authors ("good_authors"). Then, the script
prints some statistics about how the clustering performed on averaged on those
categories.
"""
def test_epsilon():
    embeddings = pkl.load(open("embeddings.pkl","rb"))
    good_authors = [0, 1, 5, 26, 37]
    bad_authors = [2,3, 8, 15, 18, 20]
    for epsilon in range(90,115,3):
        noise_frac = 0
        num_classes = 0
        class_frac = 0
        for idx in good_authors:
            cluster = DBSCAN(eps=0.01*epsilon,
                     min_samples=5,
                     n_jobs=48).fit(embeddings[idx]).labels_
            values, counts = np.unique(cluster, return_counts=True)
            if -1 in values:
                noise_frac += counts[0]/np.sum(counts)
                num_classes += values.shape[0]-1
            else:
                noise_frac += 0
                num_classes += values.shape[0]

            class_frac += np.amax(counts[values >= 0])/np.sum(counts)

        noise_frac /= len(good_authors)
        num_classes /= len(good_authors)
        class_frac /= len(good_authors)
        print("%-5.2f %4.2f %4.2f %3i " % (0.01*epsilon, 
                                           class_frac, 
                                           noise_frac, 
                                           num_classes), 
                                        end = "")
        noise_frac = 0
        num_classes = 0
        class_frac = 0
        for idx in bad_authors:
            cluster = DBSCAN(eps=0.01*epsilon,
                     min_samples=5,
                     n_jobs=48).fit(embeddings[idx]).labels_
            values, counts = np.unique(cluster, return_counts=True)
            if -1 in values:
                noise_frac += counts[0]/np.sum(counts)
                num_classes += values.shape[0]-1
            else:
                noise_frac += 0
                num_classes += values.shape[0]

            class_frac += np.amax(counts[values >= 0])/np.sum(counts)

        noise_frac /= len(bad_authors)
        num_classes /= len(bad_authors)
        class_frac /= len(bad_authors)
        print(" |  %4.2f %4.2f %3i " % (class_frac, noise_frac, num_classes))


""" Clustering Algorithm
The feature embeddings are read from the disk and DBSCAN clustering is 
performed. The results are written back to the disk.
"""
def cluster_abstracts():
    embeddings = pkl.load(open("embeddings.pkl","rb"))
    clusters = []
    for idx in range(0,len(embeddings)):
        cluster = DBSCAN(eps=0.99,
                 min_samples=5,
                 n_jobs=48).fit(embeddings[idx]).labels_
        clusters.append(cluster)
        if not (idx % 100):
            print(idx)

    pkl.dump(clusters, open("clusters.pkl", "wb"))


embed()
#  test_epsilon()
cluster_abstracts()

""" In a final step, the clustering results are used to reject ambiguous authors.
An author is classified as valid/non_ambiguous if:
    - the DBSCAN found only 1 class + noise 
    OR 
    - the main class makes up 85% of the data no-noise data AND
    - there is less than 15% classified as noise 
Those parameters are hand-tuned as explained in the Summary.
"""
clusters = pkl.load(open("clusters.pkl", "rb"))
is_single_author = []
with open("dataset/authors.txt") as f:
    authors = f.read().strip("\n").split("\n")

for i in range(len(clusters)):
    cluster = clusters[i]
    values, counts = np.unique(cluster, return_counts=True)
    if -1 in values:
        noise_frac = counts[0]/np.sum(counts)
        num_classes = values.shape[0]-1
    else:
        noise_frac = 0
        num_classes = values.shape[0]
    if len(values) == 1 and values[0] == [-1] :
        is_single_author.append(False)
        continue
    class_frac = np.amax(counts[values >= 0])/np.sum(counts)
    if (num_classes == 1 or (class_frac > 0.85 and noise_frac < 0.15)):
        is_single_author.append(True)
    else:
        is_single_author.append(False)
    print("%-20s %3s %4.2f %4.2f %3i %4i" % (
        authors[i],
        ("Yes" if is_single_author[i] else "No"),
        class_frac,
        noise_frac,
        num_classes,
        sum(counts)))

# Write everything to a file
bad_authors = []
for (a, m) in zip(authors, is_single_author):
    if not m:
        bad_authors.append(a)

with open("dataset/bad_authors.csv", "w") as f:
    f.write("\n".join(bad_authors) + "\n")
