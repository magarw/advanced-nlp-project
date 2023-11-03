# advanced-nlp-project

This repository contains all the code we used to reproduce:
> Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. 2017. Enriching Word Vectors with Subword Information. Transactions of the Association for Computational Linguistics, 5:135–146.

Code is located in code/ along with anothe README that will guide you on the steps to follow to run our experiments, along with expected time for each step. Since we reimplemented the codebase and retrained almost all the embeddings, that step naturally took the longest, and evaluation also took 1-2 days. 
The data folder contains the test data, but doesn't include any training data. For that, you'll need to download the Wikipedia dumps from the links and filenames provided in the report, as well download the fastText embeddings yourself since that will take over 100-200GB on your laptop.
