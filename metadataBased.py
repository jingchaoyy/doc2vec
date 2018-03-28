""" Analysis metadata of datasets with doc2vec,
each metadata has been saved as a single json file,
including dataset name, descriptions, etc.
Author: Jingchao Yang
Date: Mar 27 2018
"""

from os import listdir
import json
import gensim

LabeledSentence = gensim.models.doc2vec.LabeledSentence
from os.path import isfile, join

""" Doc2Vec is using two things when training your model, 
labels and the actual data. The labels can be anything, 
but to make it easier each document file name will be its’ label.
"""
docLabels = []
# getting all the names of all json files
docLabels = [f for f in listdir("/Users/YJccccc/doc2vec/RawMetadata/") if f.endswith('.json')]
print(docLabels)

data = []  # storing target text contents all together
for doc in docLabels:
    # read each json file
    desc = json.load(open("/Users/YJccccc/doc2vec/RawMetadata/" + doc))
    # getting data for Dataset-LongName-Full and Dataset-Description
    dName = desc['Dataset-LongName-Full']
    dDesc = desc['Dataset-Description']
    # first check if json has the Dataset-Metadata key
    if desc.get('Dataset-Metadata'):
        mData = desc['Dataset-Metadata']
        dMetaArr = []
        # loop into the Dataset-Metadata and replace white space with underline
        for i in mData:
            dMetaArr.append(i.replace(" ", "_"))
        # convert list to string
        dMeta = ', '.join(dMetaArr)
    else:  # if not Dataset-Metadata key, put blank to hole a position in the list
        dMeta = " "

    data.append('Full Name: ' + dName + '\n' + 'Description: ' + dDesc + '\n' + 'Metadata: ' + dMeta)
# print(data[3])

# for cont in data:
#     print(cont,"\n")

""" Preparing the data for Gensim Doc2vec
Gensim Doc2Vec needs model training data in an LabeledSentence iterator object
"""


class LabeledLineSentence(object):
    # supply both the raw data and the list of labels
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        # loop through all the docs, and put the documents filename as a the label for each document
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words=doc.split(), tags=[self.labels_list[idx]])


"""Training the model"""
it = LabeledLineSentence(data, docLabels)  # create the iter object

model = gensim.models.Doc2Vec(vector_size=300, window=10, min_count=5, workers=11, alpha=0.025,
                              min_alpha=0.025)  # use fixed learning rate
model.build_vocab(it)
for epoch in range(10):
    model.train(it, total_examples=model.corpus_count, epochs=model.epochs)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no deca
    # model.train(it,total_examples=model.corpus_count,epochs=model.epochs)

# save the model
model.save("/Users/YJccccc/doc2vec/doc2vec.model")

# """Testing the model"""
# print (model.most_similar("Tropical"))
