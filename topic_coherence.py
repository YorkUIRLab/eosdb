
# coding: utf-8

# In[4]:

import glob
from datetime import datetime
import logging as log
import gensim
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim
from gensim.models import CoherenceModel
from sklearn.externals import joblib
import gzip
from multiprocessing import Pool
get_ipython().magic(u'matplotlib notebook')


# In[1]:

class ModelSimilarity:

    # Uses a model (e.g. Word2Vec model) to calculate the similarity between two terms.
    
    def __init__(self, model):
        self.model = model

    def similarity(self, ranking_i, ranking_j):
        sim = 0.0
        pairs = 0
        for term_i in ranking_i:
            for term_j in ranking_j:
                try:
                    sim += self.model.similarity(term_i, term_j)
                    pairs += 1
                except:
                    # print "Failed pair (%s,%s)" % (term_i,term_j)
                    pass
        if pairs == 0:
            return 0.0
        return sim / pairs


# In[2]:

class WithinTopicMeasure:
 
    # Measures within-topic coherence for a topic model, based on a set of term rankings.

    def __init__(self, metric):
        self.metric = metric

    def evaluate_ranking(self, term_ranking):
        return self.metric.similarity(term_ranking, term_ranking)

    def evaluate_rankings(self, term_rankings):
        scores = []
        overall = 0.0
        for topic_index in range(len(term_rankings)):
            score = self.evaluate_ranking(term_rankings[topic_index])
            scores.append(score)
            overall += score
        overall /= len(term_rankings)
        return overall


# In[13]:

# To get the topic words from the model
def get_topics(ldamodel, num_topics, num_words):
    topics = []
    for topic_id, topic in ldamodel.show_topics(num_topics=num_topics, num_words=num_words, formatted=False):
        topic = [word for word, _ in topic]
        topics.append(topic)
    return topics

ldamodel = joblib.load('data/eos/lda/28_LDAmodel_EOS.pkl') 
print(ldamodel)
print(get_topics(ldamodel, 28, 10))


# In[18]:

model_path = 'data/eos/word2vec_model_all.model'
log.info("Loading Word2Vec model from %s ..." % model_path)

model = gensim.models.Word2Vec.load(model_path)

metric = ModelSimilarity(model)
validation_measure = WithinTopicMeasure(metric)

topic_num = 28
truncated_term_rankings = get_topics(ldamodel, topic_num, 10)
coherence_scores = validation_measure.evaluate_rankings(truncated_term_rankings)
log.info("Model coherence (k=%d) = %.4f" % (topic_num, coherence_scores))
print(coherence_scores)


# In[ ]:



