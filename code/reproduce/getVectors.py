import gensim
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts
from gensim.models import Word2Vec


def getVectors_for_5_1_data(bin, test_name, lang):
    print('loading',bin,'...')
    model = gensim.models.KeyedVectors.load(bin)
    print('gotcha!')

    file = open(test_name+'.txt','r')
    new_vects = open('embed_output_5-1-'+lang+'-'+test_name+'.txt','w')
    new_gold = open('gold_input_5-1-'+lang+'-'+test_name+'.txt','w')
    f = file.readlines()
    file.close()
    words = list()
    for line in f:
        if line != '':
            l = line.split('\t')
            word1 = l[0]
            word2 = l[1]
            avg = l[2]
            # check that word exists, then write to the files
            # word 1
            if word1 in model:
                new_vects.write(word1+' ')
                for num in model[word1]:
                    new_vects.write(str(num)+' ')
                new_vects.write('\n')
            else:
                new_vects.write(word1+' ')
                for i in range(300):
                    new_vects.write('0.0 ')
                new_vects.write('\n')
            # word 2
            if word2 in model:
                new_vects.write(word2+' ')
                for num in model[word2]:
                    new_vects.write(str(num)+' ')
                new_vects.write('\n')
            else:
                new_vects.write(word2+' ')
                for i in range(300):
                    new_vects.write('0.0 ')
                new_vects.write('\n')
            # write to the gold file
            new_gold.write(word1+'\t'+word2+'\t'+avg+'\n')
    new_gold.close()
    new_vects.close()

def format_German(file, new_name):
    f = open(file,'r')
    lines = f.readlines()
    new_lines = list()
    for line in lines:
        l = line.split(':')
        new_lines.append('\t'.join(l))
    f.close()
    
    f = open(new_name,'w')
    for line in new_lines:
        f.write(line.lower())
    f.close()
    
    


getVectors_for_5_1_data('EnModel.bin','rw','en')


