
def get_cosine_scores(filename):

    from scipy.spatial.distance import cosine

    with open(filename, "r") as f:
        embeddings = f.readlines()

    cosine_scores = [ ]
    num = 0
    overall_extracted_embeddings = [ ]
    overall_extracted_words = [ ]
    for line in embeddings:
        line = line.strip()
        word = line[:line.index(" ")]

        em = [float(i) for i in line[line.index(" "):].strip().split(" ")]

        overall_extracted_words.append(word)
        overall_extracted_embeddings.append(em)
        num += 1
        if num %2 == 0:
            cosine_scores.append((overall_extracted_words[0], overall_extracted_words[1], 1 - cosine(overall_extracted_embeddings[0], overall_extracted_embeddings[1])  ))
            overall_extracted_words = [ ]
            overall_extracted_embeddings = [ ]

    return cosine_scores
def get_reference_scores(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        data = [y.split("\t") for y in lines]
        if len(data[0]) != 3:
            # print(data[0])
            data = [y.split(" ") for y in lines]
            # print(data[0])
        for i in range(len(data)):
            x = data[i]
            if len(x) != 3:
                print('x')
                continue
            data[i] = (x[0], x[1], float(x[2].strip("\n")))

    return data
def spearman_score(l1, l2):
    from scipy import stats
    assert len(l1) == len(l2)
    l1 = [x[2] for x in l1]
    l2 = [x[2] for x in l2]
    res = stats.spearmanr(l1, l2)
    print(res)
    print(res.statistic)
def preprocess_rw():
    with open("../data/rw.txt", "r") as f:
        data = f.readlines()
    overall_data = [ ]
    for i in range(len(data)):
        line = data[i].split("\t")
        overall_data.append((line[0], line[1], line[2]))
    source_words = [ ]
    for i in range(len(overall_data)):
        source_words.append(overall_data[i][0])
        source_words.append(overall_data[i][1])

    with open("../data/em-out/embed_source_5-1-eng-rw.txt", "w") as f:
        for x in source_words:
            f.write(x + "\n")
    with open("../data/em-out/embed_source_scores_5-1-eng-rw.txt", "w") as f:
        for x in overall_data:
            f.write(f"{x[0]}\t{x[1]}\t{x[2]}\n")
def preprocess_ws353():
    import pandas as pd
    x = pd.read_csv("../data/WS353.csv", sep=";")
    print(x.head())
    english = x.iloc[:,[0,1,-1]]
    romanian =x.iloc[:,[2,3,-1]]
    arabic =x.iloc[:,[4,5,-1]]
    spanish =x.iloc[:,[6,7,-1]]

    with open("../data/gold-inputs/gold_input_5-1-en-WS353.txt", "w") as f:
        source_words = []
        for i in range(english.shape[0]):
            f.write(f"{english.iloc[i,0]} {english.iloc[i,1]} {english.iloc[i,2]}\n")
            source_words.append(english.iloc[i,0])
            source_words.append(english.iloc[i,1])

        with open("../data/en-WS353_5-1-words.txt", "w") as f:
            for x in source_words:
                f.write(x + "\n")

    with open("../data/gold-inputs/gold_input_5-1-ro-WS353.txt", "w") as f:
        source_words = []
        for i in range(english.shape[0]):
            f.write(f"{romanian.iloc[i,0]} {romanian.iloc[i,1]} {romanian.iloc[i,2]}\n")
            source_words.append(romanian.iloc[i,0])
            source_words.append(romanian.iloc[i,1])
        with open("../data/ro-WS353_5-1-words.txt", "w") as f:
            for x in source_words:
                f.write(x + "\n")

    with open("../data/gold-inputs/gold_input_5-1-ar-WS353.txt", "w") as f:
        source_words = []
        for i in range(english.shape[0]):
            f.write(f"{arabic.iloc[i,0]} {arabic.iloc[i,1]} {arabic.iloc[i,2]}\n")
            source_words.append(arabic.iloc[i,0])
            source_words.append(arabic.iloc[i,1])
        with open("../data/ar-WS353_5-1-words.txt", "w") as f:
            for x in source_words:
                f.write(x + "\n")

    with open("../data/gold-inputs/gold_input_5-1-es-WS353.txt", "w") as f:
        source_words = []
        for i in range(english.shape[0]):
            f.write(f"{spanish.iloc[i,0]} {spanish.iloc[i,1]} {spanish.iloc[i,2]}\n")
            source_words.append(spanish.iloc[i,0])
            source_words.append(spanish.iloc[i,1])
        with open("../data/es-WS353_5-1-words.txt", "w") as f:
            for x in source_words:
                f.write(x + "\n")

    with open("../data/em-out/embed_source_5-1-eng-rw.txt", "w") as f:
        for x in source_words:
            f.write(x + "\n")
def preprocess_hj():
    import pandas as pd
    x = pd.read_csv("../data/hj.csv", sep=",")
    print(x.head())
    russian = x.iloc[:,[0,1,-1]]
    with open("../data/gold-inputs/gold_input_5-1-ru-HJ.txt", "w") as f:
        source_words = []
        for i in range(russian.shape[0]):
            f.write(f"{russian.iloc[i,0]} {russian.iloc[i,1]} {russian.iloc[i,2]}\n")
            source_words.append(russian.iloc[i,1])
            source_words.append(russian.iloc[i,0])

        with open("../data/ru-HJ_5-1-words.txt", "w") as f:
            for x in source_words:
                f.write(x + "\n")
def preprocess_defr_5_1():
    with open("../data/gold-inputs/gold_input_5-1-de-GUR65-GERMAN-gold.txt", "r") as f:
        lines = f.readlines()
        d = [x.split("\t") for x in lines]
        if len(d[0]) != 3:
            d = [x.split(" ") for x in lines]
            if len(d[0]) != 3:
                print("Error")
        with open("../data/de-GUR65_5-1-words.txt", "w") as f2:
            for i in d:
                f2.write(i[0] + "\n")
                f2.write(i[1] + "\n")

    with open("../data/gold-inputs/gold_input_5-1-de-GUR350-GERMAN-gold.txt", "r") as f:
        lines = f.readlines()
        d = [x.split("\t") for x in lines]
        if len(d[0]) != 3:
            d = [x.split(" ") for x in lines]
            if len(d[0]) != 3:
                print("Error")
        with open("../data/de-GUR350_5-1-words.txt", "w") as f2:
            for i in d:
                f2.write(i[0] + "\n")
                f2.write(i[1] + "\n")

    with open("../data/gold-inputs/gold_input_5-1-de-ZG222-GERMAN-gold.txt", "r") as f:
        lines = f.readlines()
        d = [x.split("\t") for x in lines]
        if len(d[0]) != 3:
            d = [x.split(" ") for x in lines]
            if len(d[0]) != 3:
                print("Error")
        with open("../data/de-ZG222_5-1-words.txt", "w") as f2:
            for i in d:
                f2.write(i[0] + "\n")
                f2.write(i[1] + "\n")

    with open("../data/gold-inputs/gold_input_5-1-fr-RG65_FR.txt", "r") as f:
        lines = f.readlines()
        d = [x.split("\t") for x in lines]
        if len(d[0]) != 3:
            d = [x.split(" ") for x in lines]
            if len(d[0]) != 3:
                print("Error")
        with open("../data/fr-RG65_5-1-words.txt", "w") as f2:
            for i in d:
                f2.write(i[0] + "\n")
                f2.write(i[1] + "\n")
def check_vec(reference_scores, lang, dataset):
    import io
    look_words_embeddings = {}
    look_words = set()
    word_order = [ ]
    for each_tuple in reference_scores:
        look_words_embeddings[each_tuple[0]] = 0
        look_words_embeddings[each_tuple[1]] = 0
        look_words.add(each_tuple[0])
        look_words.add(each_tuple[1])
        word_order.append(each_tuple[0])
        word_order.append(each_tuple[1])

    input_file = io.open(f"../models/wiki.{lang}/wiki.{lang}.vec", 'r', encoding='utf-8', newline='\n', errors='ignore')

    no_of_words, vector_size = map(int, input_file.readline().split())
    print("num words in training", no_of_words) # 2519370 words

    i = 0
    for line in input_file.readlines():
        i = i + 1
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in look_words:
            look_words_embeddings[word]  = list(map(float, tokens[1:]))
        if i % 100000 == 0:
            print(i,no_of_words )

    embeddings = [ ]
    for w in word_order:
        embeddings.append((w, look_words_embeddings[w]))

    with open(f"../data/em-out/embed_output_noOOV_5-1-{lang}-{dataset}.txt", 'w') as f:
        for t in embeddings:
            x = list(t)
            if x[1] == 0:
                x[1] = [0]*300

            f.write(f"{x[0]} {' '.join(str(i) for i in x[1])}\n")

    return None

def cbow_test(reference_scores, lang, dataset):
    from gensim.models.keyedvectors import KeyedVectors
    import io

    word_order = [ ]
    look_words_embeddings = {}
    look_words = set()
    for each_tuple in reference_scores:
        look_words_embeddings[each_tuple[0]] =[0]*300
        look_words_embeddings[each_tuple[1]] = [0]*300
        look_words.add(each_tuple[0])
        look_words.add(each_tuple[1])
        word_order.append(each_tuple[0])
        word_order.append(each_tuple[1])

    wordvecs = KeyedVectors.load(f"../data/Embeddings-CBOW/{lang}wiki-word2vec.wordvectors", mmap='r')
    print("word vectors loaded")
    embeddings = [ ]
    for word in word_order:
        try:
            look_words_embeddings[word]  = wordvecs[word]
        except:
            pass
        embeddings.append((word, look_words_embeddings[word]))
    del wordvecs

    with open(f"../data/em-out/embed_output_CBOW_5-1-{lang}-{dataset}.txt", "w") as f:
        for t in embeddings:
            x = list(t)
            f.write(f"{x[0]} {' '.join(str(i) for i in x[1])}\n")

    cosine_scores_CBOW = get_cosine_scores(f"../data/em-out/embed_output_CBOW_5-1-{lang}-{dataset}.txt")
    spearman_score(reference_scores, cosine_scores_CBOW)
def skipgram_test(reference_scores, lang, dataset):
    from gensim.models.keyedvectors import KeyedVectors
    import io

    word_order = [ ]
    look_words_embeddings = {}
    look_words = set()
    for each_tuple in reference_scores:
        look_words_embeddings[each_tuple[0]] =[0]*300
        look_words_embeddings[each_tuple[1]] = [0]*300
        look_words.add(each_tuple[0])
        look_words.add(each_tuple[1])
        word_order.append(each_tuple[0])
        word_order.append(each_tuple[1])

    wordvecs = KeyedVectors.load(f"../data/Embeddings-Skipgram/{lang}wiki-sg-word2vec.wordvectors", mmap='r')
    print("word vectors loaded")
    embeddings = [ ]
    for word in word_order:
        try:
            look_words_embeddings[word]  = wordvecs[word]
        except:
            pass
        embeddings.append((word, look_words_embeddings[word]))
    del wordvecs

    with open(f"../data/em-out/embed_output_SKIPGRAM_5-1-{lang}-{dataset}.txt", "w") as f:
        for t in embeddings:
            x = list(t)
            f.write(f"{x[0]} {' '.join(str(i) for i in x[1])}\n")

    cosine_scores_CBOW = get_cosine_scores(f"../data/em-out/embed_output_SKIPGRAM_5-1-{lang}-{dataset}.txt")
    spearman_score(reference_scores, cosine_scores_CBOW)

# preprocess()
# preprocess_ws353()
# preprocess_hj()
# preprocess_defr_5_1()

print("CBOW Results - Table 5.1")
print("Russian HJ")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-ru-HJ.txt")
cbow_test(reference_scores, 'ru', 'HJ')

print("German GUR65")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-de-GUR65-GERMAN-gold.txt")
cbow_test(reference_scores, 'de', 'GUR65')

print("German GUR350")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-de-GUR350-GERMAN-gold.txt")
cbow_test(reference_scores, 'de', 'GUR350')

print("German ZG222")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-de-ZG222-GERMAN-gold.txt")
cbow_test(reference_scores, 'de', 'ZG222')

print("Spanish WS353")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-es-WS353.txt")
cbow_test(reference_scores, 'es', 'WS353')

print("English RW")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-eng-rw.txt")
cbow_test(reference_scores, 'en', 'RW')

print("English WS353")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-en-WS353.txt")
cbow_test(reference_scores, 'en', 'WS353')

print("French RG65")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-fr-RG65_FR.txt")
cbow_test(reference_scores, 'fr', 'RG65')

print("Romanian WS353")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-ro-WS353.txt")
cbow_test(reference_scores, 'ro', 'WS353')

print("Arabic WS353")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-ar-WS353.txt")
cbow_test(reference_scores, 'ar', 'WS353')



SKIPGRAM
print("Skipgram")
print("German GUR65")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-de-GUR65-GERMAN-gold.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-de-GUR65-GERMAN-gold.txt")
spearman_score(reference_scores, cosine_scores)

print("German GUR350")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-de-GUR350-GERMAN-gold.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-de-GUR350-GERMAN-gold.txt")
spearman_score(reference_scores, cosine_scores)

print("German ZG222")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-de-ZG222-GERMAN-gold.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-de-ZG222-GERMAN-gold.txt")
spearman_score(reference_scores, cosine_scores)

print("Spanish WS353")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-es-WS353.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-es-WS353-SPANISH.txt")
spearman_score(reference_scores, cosine_scores)

print("Arabic WS353")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-ar-WS353.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-ar-WS353-ARABIC.txt")
spearman_score(reference_scores, cosine_scores)

print("Russian HJ")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-ru-HJ.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-HJ-RUSSIAN.txt")
spearman_score(reference_scores, cosine_scores)

print("English ")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-en-WS353.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-en-WS353-ENGLISH.txt")
spearman_score(reference_scores, cosine_scores)

ROMANIAN
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-ro-WS353.txt")
skipgram_test(reference_scores, 'ro', 'WS353')


print("English ")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-eng-rw.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-en-rw.txt")
spearman_score(reference_scores, cosine_scores)


print("fasttext")
print("English RW")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-eng-rw.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-eng-rw.txt")
spearman_score(reference_scores, cosine_scores)
check_vec(reference_scores, 'en', 'rw')
cosine_scores_no_OOV = get_cosine_scores("../data/em-out/embed_output_noOOV_5-1-eng-rw.txt")
spearman_score(reference_scores, cosine_scores_no_OOV)

print("Russian HJ")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-ru-HJ.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-ru-HJ.txt")
spearman_score(reference_scores, cosine_scores)
check_vec(reference_scores, 'ru', 'HJ')
cosine_scores_no_OOV = get_cosine_scores("../data/em-out/embed_output_noOOV_5-1-ru-HJ.txt")
spearman_score(reference_scores, cosine_scores_no_OOV)


print("French RG65")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-fr-RG65_FR.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-fr-RG65_FR.txt")
spearman_score(reference_scores, cosine_scores)
check_vec(reference_scores, 'fr', 'RG65')
cosine_scores_no_OOV = get_cosine_scores("../data/em-out/embed_output_noOOV_5-1-fr-RG65.txt")
spearman_score(reference_scores, cosine_scores_no_OOV)


print("Spanish WS353")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-es-WS353.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-es-WS353.txt")
spearman_score(reference_scores, cosine_scores)
check_vec(reference_scores, 'es', 'WS353')
cosine_scores_no_OOV = get_cosine_scores("../data/em-out/embed_output_noOOV_5-1-es-WS353.txt")
spearman_score(reference_scores, cosine_scores_no_OOV)


print("English WS353")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-en-WS353.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-en-WS353.txt")
spearman_score(reference_scores, cosine_scores)
check_vec(reference_scores, 'en', 'WS353')
cosine_scores_no_OOV = get_cosine_scores("../data/em-out/embed_output_noOOV_5-1-en-WS353.txt")
spearman_score(reference_scores, cosine_scores_no_OOV)

print("Romanian WS353")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-ro-WS353.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-ro-WS353.txt")
spearman_score(reference_scores, cosine_scores)
check_vec(reference_scores, 'ro', 'WS353')
cosine_scores_no_OOV = get_cosine_scores("../data/em-out/embed_output_noOOV_5-1-ro-WS353.txt")
spearman_score(reference_scores, cosine_scores_no_OOV)


print("Arabic WS353")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-ar-WS353.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-ar-WS353.txt")
spearman_score(reference_scores, cosine_scores)
check_vec(reference_scores, 'ar', 'WS353')
cosine_scores_no_OOV = get_cosine_scores("../data/em-out/embed_output_noOOV_5-1-ar-WS353.txt")
spearman_score(reference_scores, cosine_scores_no_OOV)

print("German GUR65")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-de-GUR65-GERMAN-gold.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-de-GUR65-GERMAN-gold.txt")
spearman_score(reference_scores, cosine_scores)
check_vec(reference_scores, 'de', 'GUR65')
cosine_scores_no_OOV = get_cosine_scores("../data/em-out/embed_output_noOOV_5-1-de-GUR65.txt")
spearman_score(reference_scores, cosine_scores_no_OOV)


print("German GUR350")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-de-GUR350-GERMAN-gold.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-de-GUR350-GERMAN-gold.txt")
spearman_score(reference_scores, cosine_scores)
check_vec(reference_scores, 'de', 'GUR350')
cosine_scores_no_OOV = get_cosine_scores("../data/em-out/embed_output_noOOV_5-1-de-GUR350.txt")
spearman_score(reference_scores, cosine_scores_no_OOV)

print("German ZG222")
reference_scores = get_reference_scores("../data/gold-inputs/gold_input_5-1-de-ZG222-GERMAN-gold.txt")
cosine_scores = get_cosine_scores("../data/em-out/embed_output_5-1-de-ZG222-GERMAN-gold.txt")
spearman_score(reference_scores, cosine_scores)
check_vec(reference_scores, 'de', 'ZG222')
cosine_scores_no_OOV = get_cosine_scores("../data/em-out/embed_output_noOOV_5-1-de-ZG222.txt")
spearman_score(reference_scores, cosine_scores_no_OOV)
