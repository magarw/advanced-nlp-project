from gensim.models.word2vec import BrownCorpus, Word2Vec, LineSentence
from gensim.test.utils import datapath

import logging
import multiprocessing
from pprint import pprint
import smart_open
from gensim.corpora.wikicorpus import WikiCorpus, tokenize

"""
Romanian: 442,068 articles (600MB) [[241000]]
Czech:    533,419 articles (1GB) [[491000]]
Arabic:   1,220,823 articles (1.5GB) [[876000]]
Italian:  1,832,909 articles (3.63GB)
Spanish:  1,899,000 articles (4.29GB)
Russian:  1,942,000 articles (5.14GB)
French:   2,564,579 articles (5.93GB)
German:   2,847,750 articles (6.84GB)
English:  6,737,260 articles (21.42GB)
"""

import os
if __name__ == '__main__':
    NAME = "enwiki-20231020-pages-articles"
    if not os.path.exists(f"../data/wiki_dumps/TXT/{NAME}.txt.gz"):
        FILEPATH = f"../data/wiki_dumps/{NAME}.xml.bz2"
        wiki = WikiCorpus(
            FILEPATH,  # path to the file you downloaded above
            tokenizer_func=tokenize,  # simple regexp; plug in your own tokenizer here
            metadata=True,  # also return the article titles and ids when parsing
            dictionary={},  # don't start processing the data yet
        )


        with smart_open.open(f"../data/wiki_dumps/TXT/{NAME}.txt.gz", "w", encoding='utf8') as fout:
            for article_no, (content, (page_id, title)) in enumerate(wiki.get_texts()):
                title = ' '.join(title.split())
                if article_no % 1000 == 0:
                    print(f"processing article #{article_no}: {title} ({len(content)} tokens)" )
                fout.write(f"{title}\t{' '.join(content)}\n")  # title_of_article [TAB] words of the article

    else:
        import random
        data = LineSentence(f"../data/wiki_dumps/TXT/{NAME}.txt.gz")
        print("Data loaded")
        # CBOW Baseline
        # model = Word2Vec(vector_size=300,window=random.uniform(1, 5),min_count=5,sg=0,negative=5,alpha=0.05)
        # Skipgram
        model = Word2Vec(vector_size=300,window=random.uniform(1, 5),min_count=5,sg=1,negative=5,alpha=0.025)
        print("Model initialized")

        model.build_vocab(data)
        print("Vocabulary built")
        model.train(data, total_examples=model.corpus_count, epochs=model.epochs)
        print("Training")
        word_vectors = model.wv
        del model
        word_vectors.save("enwiki-sg-word2vec.wordvectors")
        print("word vectors saved")



    #
