""" Analysis user patterns (log files) with doc2vec,
each user behavior has been saved as a single json file,
including user queries, datasets, temporal resolution, etc.
Author: Jingchao Yang
Date: Mar 27 2018
"""

from os import listdir
import json
import gensim

LabeledSentence = gensim.models.doc2vec.LabeledSentence

""" Doc2Vec is using two things when training your model, 
labels and the actual data. The labels can be anything, 
but to make it easier each document file name will be its’ label.
"""
docLabels = []
# getting all the names of all json files
docLabels = [f for f in listdir("/Users/YJccccc/doc2vec/userData/") if f.endswith('.json')]
print(docLabels)

data = []
for doc in docLabels:
    # read each json file
    userData = json.load(open("/Users/YJccccc/doc2vec/userData/" + doc))
    # getting data for only Dataset-Description
    # sessionID.append(userData['sessionId'])

    clicks = userData["click"]
    query, datasets, platform, tResolution, measur = [], [], [], [], []
    # loop into "click" and collect all user queries and clicked datasets
    for i in range(0, len(clicks)):
        query.append(clicks[i]['query'])
        datasets.append(clicks[i]['dataset'])
        # check if there has the platform key under click

        if clicks[i].get('platform'):
            platform.append(clicks[i]['platform'])
        else:
            platform.append(" ")

        # check if there has the temporalresolution key under click
        if clicks[i].get('temporalresolution'):
            tList = []
            tList.append(clicks[i]['temporalresolution'])
            # loop into the temporalresolution and replace white space with underline
            for j in tList:
                tResolution.append(j.replace(" ", "_"))
        else:
            tResolution.append(" ")

        # check if there has the measurement key under click
        if clicks[i].get('measurement'):
            tList = []
            tList.append(clicks[i]['measurement'])
            # loop into the measurement and replace white space with underline
            for j in tList:
                measur.append(j.replace(" ", "_"))
        else:
            measur.append(" ")

    data.append('User Query: ' + ' '.join(query) + '\n' + 'Datasets: ' + ' '.join(datasets)
                + '\n' + 'Platform: ' + ' '.join(platform) + '\n' + 'tResolution: ' + ' '.join(tResolution)
                + '\n' + 'Measure: ' + ' '.join(measur))

# print(data[1])

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
            yield LabeledSentence(words=doc.split(),tags=[self.labels_list[idx]])

"""Training the model"""
it = LabeledLineSentence(data, docLabels)  # create the iter object

model = gensim.models.Doc2Vec(vector_size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning rate
model.build_vocab(it)
for epoch in range(10):
    model.train(it,total_examples=model.corpus_count,epochs=model.epochs)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    # model.train(it,total_examples=model.corpus_count,epochs=model.epochs)

# save the model
model.save("/Users/YJccccc/doc2vec/doc2vec_userdata.model")

"""Testing the model"""
# print (model.most_similar("Tropical"))
