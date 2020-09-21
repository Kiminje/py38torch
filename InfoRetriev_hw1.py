########################################################
#######B515031 김인제 information retrieval HW1###########
###################2020. 09. 21.########################
from gensim.models.word2vec import Text8Corpus
import gensim.models.word2vec as gs

# 1. please change to appropriate directory
sentences = Text8Corpus('/home/inje/anaconda3/\
envs/py37torch/lib/python3.8/site-packages/gensim/models/text8')
#   if using clone use the below
#   sentences = Text8Corpus('text8')

for sentence in sentences:
    #print(sentence[0:10]) #    uncomment to activate #1
    break

# 2. model generate
#model = gs.Word2Vec(sentences)
#print(model)    # generated model      #    uncomment to activate #2
#  3. model save & load
#model.save("inje.model")    # model save       #    uncomment to activate #3
model = gs.Word2Vec.load("inje.model")  # model load
#print(model)    # loaded model     #    uncomment to activate #3
#  4. type of model.wv.vocab
#print(type(model.wv.vocab))        #    uncomment to activate #4
#  5. print model.wv.vocab
#print(model.wv.vocab)              #    uncomment to activate #5
#  6. print word2vec's words (not need to print all of them.)
#print(list(model.wv.vocab.keys())[0:10])       #    uncomment to activate #6
#  7. word vector of 'computer'
#print("word vector of 'computer'\n", model.wv['computer'], "\n size is {}".format(model.wv['computer'].size))      #    uncomment to activate #7
#  8. the most similar word with 'computer'
#print("the most similar word with 'computer' is\n",model.wv.similar_by_vector('computer')) #    uncomment to activate #8
#  9. find most similar word (most similar = woman + king - man)
print("\n***************find most similar word of 'woman + king - man'********************")
MostSimilar = model.wv['woman'] + model.wv['king'] - model.wv['man']
print(MostSimilar)
print("most similar key word:\n", model.wv.similar_by_vector(MostSimilar))


print("\n*****************EXTRA****************\n")
from nltk.corpus import wordnet as wn
q = wn.synsets('computer')[0].definition()

def definition(word):
    return wn.synsets(word)[0].definition()

q = definition('computer')
#   split the definition sentence of 'computer' to word by word.
#   rating 50 ranks for most similar word of 'computer'
Result = model.wv.most_similar(positive=q.split(), topn=50)
print(Result)
#if 'computer' detected in rank, print the key
for key, item in Result:
    if 'computer' in key:
        print(key)
else:
    print("finish")
