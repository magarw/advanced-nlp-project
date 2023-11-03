# This code downloads the necessary Word2Vec Wikipedia-trained models in multiple languages

from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download

# English
# https://huggingface.co/Word2vec/wikipedia2vec_enwiki_20180420_300d
print('loading English Wikipedia vectors...')
model = KeyedVectors.load_word2vec_format('enwiki_20180420_300d.txt')#hf_hub_download(repo_id="Word2vec/wikipedia2vec_enwiki_20180420_300d", filename="enwiki_20180420_300d.txt"))
print('saving...')
model.save("EnModel.bin")
print('saved')
'''
# French
# https://huggingface.co/Word2vec/wikipedia2vec_frwiki_20180420_300d
print('downloading French...')
model = KeyedVectors.load_word2vec_format(hf_hub_download(repo_id="Word2vec/wikipedia2vec_frwiki_20180420_300d", filename="frwiki_20180420_300d.txt"))
print('saving...')
model.save("FrModel.bin")
print('saved')

# German
# https://huggingface.co/Word2vec/wikipedia2vec_dewiki_20180420_300d
print('downloading German...')
model = hf_hub_download(repo_id="Word2vec/wikipedia2vec_dewiki_20180420_300d", filename="dewiki_20180420_300d.txt"))
print('saving...')
model.save("DeModel.bin")
print('saved')


# Russian
# https://huggingface.co/Word2vec/wikipedia2vec_ruwiki_20180420_300d
print('downloading Russian...')
model = KeyedVectors.load_word2vec_format(hf_hub_download(repo_id="Word2vec/wikipedia2vec_ruwiki_20180420_300d", filename="ruwiki_20180420_300d.txt"))
print('saving...')
model.save("RuModel.bin")
print('saved')


# Italian
# https://huggingface.co/Word2vec/wikipedia2vec_itwiki_20180420_300d
print('downloading Italian...')
model = KeyedVectors.load_word2vec_format(hf_hub_download(repo_id="Word2vec/wikipedia2vec_itwiki_20180420_300d", filename="itwiki_20180420_300d.txt"))
print('saving...')
model.save("ItModel.bin")
print('saved')

# Spanish
# https://huggingface.co/Word2vec/wikipedia2vec_eswiki_20180420_300d
print('downloading Spanish...')
model = KeyedVectors.load_word2vec_format(hf_hub_download(repo_id="Word2vec/wikipedia2vec_eswiki_20180420_300d", filename="eswiki_20180420_300d.txt"))
print('saving...')
model.save("EsModel.bin")
print('saved')

# Arabic
# https://huggingface.co/Word2vec/wikipedia2vec_arwiki_20180420_300d
print('downloading Arabic...')
model = KeyedVectors.load_word2vec_format(hf_hub_download(repo_id="Word2vec/wikipedia2vec_arwiki_20180420_300d", filename="arwiki_20180420_300d.txt"))
print('saving...')
model.save("ArModel.bin")
print('saved')

