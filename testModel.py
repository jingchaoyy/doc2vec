import gensim

# load the model back
model = gensim.models.Doc2Vec.load('/Users/YJccccc/doc2vec/doc2vec.model')
"""Testing the model"""
# getting vector expression of a document
docvec = model.docvecs["379.json"]
# print(docvec)

# output similar documents with ranking
similar_doc = model.docvecs.most_similar("379.json")
print(similar_doc)

# print (model.most_similar("Tropical"))