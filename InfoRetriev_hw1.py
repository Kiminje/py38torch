from nltk.corpus import wordnet as wn
q = wn.synsets('computer')[0].definition()
#first hw1

def definition(word):
    return wn.synsets(word)[0].definition()

from gensim.models.word2vec import Text8Corpus


from gensim.test.utils import common_texts, get_tmpfile, datapath
#   file = open("/home/inje/nltk_data/text8", "r")
import gensim.models.word2vec as gs

#   path = get_tmpfile("text")


import gensim.downloader as api

"""sentences = gs.Text8Corpus('/home/inje/anaconda3/envs/py37torch/lib/python3.8/site-packages/gensim/models/text8')

for sentence in sentences:
    print(sentence[0:10])
    break
    """
#   model = gs.Word2Vec(sentences)
#   model.save("inje.model")
model = gs.Word2Vec.load("inje.model")
print(model)
print(type(model.wv.vocab))
# print(model.wv.vocab)
#   model.train([["hello", "world"]], total_examples=1, epochs=1)
#   vector = model.wv['computer']
print(list(model.wv.vocab.keys())[0:10])
print(model.wv['computer'])
print(model.wv.similar_by_vector('computer'))
print("\n\n\n***************find most similar word of 'woman + king - man'********************")
MostSimilar = model.wv['woman'] + model.wv['king'] - model.wv['man']
print(MostSimilar)
print(model.wv.similar_by_vector(MostSimilar))


q = definition('computer')
Result = model.wv.most_similar(positive=q.split(), topn=50)
print(Result)
for key, item in Result:
    if 'computer' in key:
        print(key)
else:
    print("nothing match with computer")