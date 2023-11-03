def format_mikolov():
    with open("../data/5-2/mikolov.txt", "r") as f:
        semantic = []
        syntactic = []
        flag = 0
        while True:
            line = f.readline()
            if len(line) == 0:
                break

            if line[0] == "/":
                continue
            elif line[0] == ":":
                flag = 0
                if ": gram" in line:
                    flag = 1
            else:
                if flag == 0:
                    semantic.append(line.strip().split(" "))
                elif flag == 1:
                    syntactic.append(line.strip().split(" "))

    with open("../data/5-2/mikolov-semantic.txt", "w") as f:
        for x in semantic:
            f.write(f"{x[0]} {x[1]} {x[2]} {x[3]}\n")

    with open("../data/5-2/mikolov-syntactic.txt", "w") as f:
        for x in syntactic:
            f.write(f"{x[0]} {x[1]} {x[2]} {x[3]}\n")
def format_svoboda():
    with open("../data/5-2/svoboda.txt", "r") as f:
        semantic = []
        syntactic = []
        flag = 0
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            elif line[0] == ":":
                flag = 0
                if ": gram" in line:
                    flag = 1
            else:
                if flag == 0:
                    semantic.append(line.strip().split(" "))
                elif flag == 1:
                    syntactic.append(line.strip().split(" "))

    with open("../data/5-2/svoboda-semantic.txt", "w") as f:
        for x in semantic:
            f.write(f"{x[0]} {x[1]} {x[2]} {x[3]}\n")

    with open("../data/5-2/svoboda-syntactic.txt", "w") as f:
        for x in syntactic:
            f.write(f"{x[0]} {x[1]} {x[2]} {x[3]}\n")

# Write code to compute the Czech 4th word and accuracy

# CBOW

def cbow_test(dataset, lang):
    import io
    from gensim.models.keyedvectors import KeyedVectors

    with open(f"../data/5-2/{dataset}-semantic.txt", "r") as f:
        semantic = [x.strip().split(" ") for x in f.readlines()]
    with open(f"../data/5-2/{dataset}-syntactic.txt", "r") as f:
        syntactic = [x.strip().split(" ") for x in f.readlines()]

    print(f"Language: {lang}")
    print(f"Semantic Length: {len(semantic)}")
    print(f"Syntactic Length: {len(syntactic)}")
    # For each 3 words in semantic, we compute the vectors for them, if possible,
    # through the language's CBOW model
    wordvecs = KeyedVectors.load(f"../data/Embeddings-CBOW/{lang}wiki-word2vec.wordvectors", mmap='r')

    yes = 0
    total = 0
    expected_total = len(semantic)
    for x in semantic:
        gold_word = x[3].lower()
        try:
            predicted_word = wordvecs.most_similar(positive=[ x[2].lower(), x[1].lower()], negative=[x[0].lower()])[0][0]
        except:
            continue
        total += 1
        if predicted_word == gold_word:
            yes +=1
        if total % 100 == 0:
            print("Progress: ", total, expected_total)
            print("Current Accuracy:", yes/total)
    print(f"CBOW Semantic Accuracy for {lang}, {dataset}: {yes}, {total},{expected_total}")
    yes = 0
    total = 0
    expected_total = len(syntactic)
    for x in syntactic:
        gold_word = x[3].lower()
        try:
            predicted_word = wordvecs.most_similar(positive=[ x[2].lower(), x[1].lower()], negative=[x[0].lower()])[0][0]
        except:
            continue
        total += 1
        if predicted_word == gold_word:
            yes +=1
        if total % 100 == 0:
            print("Progress: ", total, expected_total)
            print("Current Accuracy:", yes/total)


    print(f"CBOW Syntactic Accuracy for {lang}, {dataset}: {yes}, {total},{expected_total}")
def sg_test(dataset, lang):
    import io
    from gensim.models.keyedvectors import KeyedVectors

    with open(f"../data/5-2/{dataset}-semantic.txt", "r") as f:
        semantic = [x.strip().split(" ") for x in f.readlines()]
    with open(f"../data/5-2/{dataset}-syntactic.txt", "r") as f:
        syntactic = [x.strip().split(" ") for x in f.readlines()]

    print(f"Language: {lang}")
    print(f"Semantic Length: {len(semantic)}")
    print(f"Syntactic Length: {len(syntactic)}")
    # For each 3 words in semantic, we compute the vectors for them, if possible,
    # through the language's CBOW model
    wordvecs = KeyedVectors.load(f"../data/Embeddings-Skipgram/{lang}wiki-sg-word2vec.wordvectors", mmap='r')

    yes = 0
    total = 0
    expected_total = len(semantic)
    for x in semantic:
        gold_word = x[3].lower()
        try:
            predicted_word = wordvecs.most_similar(positive=[ x[2].lower(), x[1].lower()], negative=[x[0].lower()])[0][0]
        except:
            continue
        total += 1
        if predicted_word == gold_word:
            yes +=1
        if total % 100 == 0:
            print("Progress: ", total, expected_total)
            print("Current Accuracy:", yes/total)
    print(f"SKIPGRAM Semantic Accuracy for {lang}, {dataset}: {yes}, {total},{expected_total}")
    yes = 0
    total = 0
    expected_total = len(syntactic)
    for x in syntactic:
        gold_word = x[3].lower()
        try:
            predicted_word = wordvecs.most_similar(positive=[ x[2].lower(), x[1].lower()], negative=[x[0].lower()])[0][0]
        except:
            continue
        total += 1
        if predicted_word == gold_word:
            yes +=1
        if total % 100 == 0:
            print("Progress: ", total, expected_total)
            print("Current Accuracy:", yes/total)


    print(f"SKIPGRAM Syntactic Accuracy for {lang}, {dataset}: {yes}, {total},{expected_total}")
def fasttext_test(dataset):
    import fasttext
    model = fasttext.load_model("../models/wiki.de/wiki.de.bin")

    with open(f"../data/5-2/{dataset}-semantic.txt", "r") as f:
        semantic = [x.strip().split(" ") for x in f.readlines()]
    with open(f"../data/5-2/{dataset}-syntactic.txt", "r") as f:
        syntactic = [x.strip().split(" ") for x in f.readlines()]

    yes = 0
    total = 0
    for each in semantic:
        reference = each[3].lower()
        prediction =  model.get_analogies(each[1].lower(), each[0].lower(), each[2].lower())[0][1]
        if reference == prediction:
            yes += 1
        total += 1
        print(f"Semantic Accuracy for {dataset}: {yes}, {total}")


    print(f"Semantic Accuracy for {dataset}: {yes}, {total}")

    yes = 0
    total = 0
    for each in syntactic:
        reference = each[3].lower()
        prediction =  model.get_analogies(each[1].lower(), each[0].lower(), each[2].lower())[0][1]
        if reference == prediction:
            yes += 1
        total += 1
        print(f"Syntactic Accuracy for {dataset}: {yes}, {total}")

    print(f"Syntactic Accuracy for {dataset}: {yes}, {total}")
    print("heee")


format_mikolov()
format_svoboda()

fasttext_test('german')
fasttext_test('svoboda')
fasttext_test('mikolov')
fasttext_test('berardi')

sg_test('svoboda', 'cs')
sg_test('berardi', 'it')
sg_test('mikolov', 'en')
sg_test('german', 'de')

cbow_test('svoboda', 'cs')
cbow_test('berardi', 'it')
cbow_test('mikolov', 'en')
cbow_test('german', 'de')
