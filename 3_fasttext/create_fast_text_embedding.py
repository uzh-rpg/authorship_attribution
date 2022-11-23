from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import FastText
import pandas as pd
import numpy as np

from gensim.models.callbacks import CallbackAny2Vec

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

class MonitorCallback(CallbackAny2Vec):
    def __init__(self, test_words):
        self._test_words = test_words

    def on_epoch_end(self, model):
        print("Model loss:", model.get_latest_training_loss())  # print loss
        for word in self._test_words:  # show wv logic changes
            print(model.wv.most_similar(word))

class Callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print(loss)
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        print(loss_now)
        self.epoch += 1


epoch_logger = EpochLogger()

monitor = MonitorCallback(["Boylan"])  # monitor with demo words

callback = Callback()

SAVE_PATH = '/data/storage/bauersfeld/dataset-300-split/'
data_references = SAVE_PATH + '/reference_authors.csv'
EMBED_SIZE = 100
TRAINING_EPOCHS = 100
N_REFERENCES = 5

data_path_train_split_index = SAVE_PATH + '/split_index.txt'

with open(data_path_train_split_index) as f:
    SPLIT_INDEX = int(f.readline())


with open(data_references) as myfile:
    mydata = (line for line in myfile)
    text_references = pd.DataFrame(mydata, columns=['line'])

# print(text_references)

text_references['authors_per_paper'] = ''

for idx, row in text_references.iterrows():
    author_sentence = row['line'].split("####")
    authors_list_k = []
    for sentence in author_sentence:
        authors = sentence.split("|")
        authors_list_k.extend(authors)
    text_references.at[idx, 'authors_per_paper'] = authors_list_k

print("training....")


# model = Word2Vec(sentences=training_sentences, vector_size=100, window=5, min_count=1, workers=4)
# model = FastText(sentences=training_sentences, vector_size=100, window=5, min_count=1, workers=4)
model = FastText(vector_size=EMBED_SIZE, window=3, min_count=3, sentences=text_references['authors_per_paper'].iloc[0:SPLIT_INDEX].to_list(), epochs=TRAINING_EPOCHS, workers=16, callbacks = [monitor])
# model = FastText(vector_size=VEC_SIZE, window=3, min_count=3)
# model.build_vocab(training_sentences)
# model.train(training_sentences,
#             total_examples = len(training_sentences),
#             epochs = 20,
#             # workers = 16,
#             compute_loss = True,
#             callbacks = [monitor])


print(model)
print("vocabulary length: ")
print(len(model.wv))


# Print vocabulary
# for i in range(len(model.wv)):
#     print(model.wv.index_to_key[i])

text_references['reference_embedding']=''
text_references['reference_embedding_manasi']=''

print("parsing..")

for idx, row in text_references.iterrows():
    embedding_list_k = []
    embedding_list = N_REFERENCES * EMBED_SIZE * [0]
    for k in range(min(N_REFERENCES, len(row['authors_per_paper']))):
        embedding_list[k*(EMBED_SIZE):(k+1)*EMBED_SIZE] = model.wv[row['authors_per_paper'][k]]

    for author in row['authors_per_paper']:
        # embedding_list_k.append(model.wv[author])
        string_k = " ".join(["%f" % x for x in list(model.wv[author])])
        embedding_list_k.append(string_k)

    embedding_list_k = "|".join(embedding_list_k)
    text_references.at[idx, 'reference_embedding'] = embedding_list
    text_references.at[idx, 'reference_embedding_manasi'] = embedding_list_k

list_manasi = list(text_references['reference_embedding_manasi'])

reference_embedding_list = text_references['reference_embedding'].to_list()
reference_embedding_list_manasi = text_references['reference_embedding_manasi'].to_list()

import csv

with open(SAVE_PATH + "/reference_embedding.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(reference_embedding_list)


# text_references.to_csv(SAVE_PATH + "/reference_embedding_manasi_2.csv", columns = ['reference_embedding_manasi'], index=False, header=False)


with open(SAVE_PATH + "/reference_embedding_all_authors.csv","w") as f:
    f.write("\n".join(list_manasi) + "\n")

# with open(SAVE_PATH + "/reference_embedding_manasi.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(reference_embedding_list_manasi)




# print(model.wv['Andreas'])
# print(model.wv['Boylan'])
# print(model.wv['Kolchin'])
# print(model.wv['Boylan Kolchin'])
