import os
from aksharamukha import transliterate
import random
files = os.listdir("../data/raw/dev/")

scripts = ["Deva", "Guru", "Telu", "Taml","Olck","Beng","Tibt","Gujr","Knda","Mlym","Orya","Sinh"]
# languages = ["hin", "tel", "pan", "mar"]

filter_script = [x for x in files if any([y in x for y in scripts])]
# filter_langs = [x for x in filter_script if any([y in x for y in languages])]

selected_files = filter_script
print(selected_files)

map_scripts = ["Gurmukhi","Devanagari","Telugu","Ranjana", "Tamil","Malayalam","Kannada","Kaithi",
                "Assamese", "Bengali", "Chakma", "GunjalaGondi","Gujarati", "Mahajani",
                "Modi","Newa","Oriya","Santali","Sharada","Sinhala","Urdu" ]

all_train = []
all_dev = []
DEV_SIZE = 100

for file in selected_files:
    with open("../data/raw/dev/" + file,"r") as f:
        lang_code = file.split('_')[0]
        script_code = file.split('_')[1]

        lines = f.readlines()
        random.shuffle(lines)
        print(len(lines), lang_code)

        train_data = lines[:-DEV_SIZE]
        dev_data = lines[-DEV_SIZE:]


        combined_train = []
        combined_dev = []

        for each_script in map_scripts:
            y1 = [transliterate.process('autodetect', each_script, x, nativize = True, pre_options = [], post_options = []) for x in train_data ]
            y2 = [transliterate.process('autodetect', each_script, x, nativize = True, pre_options = [], post_options = []) for x in dev_data ]
            combined_train += y1
            combined_dev += y2

        # with open("../data/upscale/"+ lang_code+".dev", "w") as f2:
        #     for eachline in combined:
        #         f2.write(eachline)

        all_train += [f"__label__{lang_code}\t{x}" for x in combined_train]
        all_dev += [f"__label__{lang_code}\t{x}" for x in combined_dev]

with open("../data/upscale/upscale.train", "w") as f2:
    for eachline in all_train:
        f2.write(eachline)

with open("../data/upscale/upscale.valid", "w") as f2:
    for eachline in all_dev:
        f2.write(eachline)
