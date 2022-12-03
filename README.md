trulia_project

Objective : Predict the time it takes for a house to be sold using its description and various covariates. Also, see the impact of degrading the description (adding errors in the text) on the model's predictive power. The predictive model used is a Cox model.

Virtual environment : r_nlp2

The .yml file generated from conda is r_nlp2.yml (Linux only?). The requirements file generated with `pip freeze` is requirements.txt. To generate the environment with conda, use `conda env create -f r_nlp2.yml`.

# Utility files :
- generate_embeddings.R: R functions to generate embeddings. Embedding methods are tf-idf (with package `tm`, although I now recommend using `quanteda`), fasttext (with `reticulate` and `fasttext` (python)) and BERT (with `reticulate` and `transformers` (python)).
- utils.R: Functions used to train model and metrics to evaluate them.

# Data
- The data is not available in this public repo. You can scrape it by yourself or contact me.

# Text data
- The data is not available in this public repo. You can scrape it by yourself or contact me.

# Important files to generate results:
- exploration.Rmd: I take a look at the data extracted from trulia, remove extreme values
- variable_selection_and_performance.Rmd: I test the package `glmnet` to see what would be the best way to use it to select variables. I also try to get a few performance measures
- variable_selection_and_performance_2.Rmd: Since the first file was more about seeing how to use glmnet, I created a new file where I use what I learned form the precedent file to regularize the Cox model. Also, since I realized there is no left-truncation and length bias in the data (see file length_bias.Rmd), I can use a "standard" Cox model. I also manually censored observations with survival time too long at a lower value to have similar survival time (see length_bias.Rmd).
- corrupting_texts.Rmd: I corrupt the text (remove pronoun/numbers/proper noun and add typos) and save the text(embeddings) in degraded_texts(embeddings/degraded_texts).
- corrupted_texts_0typos.pdf : Similar to variable_selection_and_performance_2_lowIDF15, but for cox models using the degraded texts. 0typos signify that there is 0% of typos introduced in the text (see corrupting_texts.Rmd). Performance is given with structured data + texts, and texts as only covariates.

# Results
Results are printed in the .pdf files from the important files and then put in a table in [table generator](https://www.tablesgenerator.com/). The tables are then saved in the folder results/latex_generator_tables.

# Other files:
- parallel_processing_transformers.Rmd: I tried to do parallel computation (on CPU) of my function to generate word embeddings with transformers, because it takes a long time to generate the embeddings for the whole dataset (since some descriptions have >512 tokens). Unfortunately, it is not possible in R because objects created from python functions are not exportable to CPU clusters. If I try to import the transformer model in every cluster, I don't have enough RAM to generate the embeddings. Therefore, parallel computation of transformer embeddings should be done directly in python, and not in R using reticulate.
- length_bias.Rmd: I tried to see if there is a significant difference in the survival of the houses sold vs censored. In the end, we won't use any results in there, but I leave the code accessible for future reference.
- umap_test.Rmd: generated a 2D plot of transformer embeddings using UMAP instead of t-sne or PCA to see if there would be significant differences. In the end, there isn't.
- variable_selection_and_performance_2_lowerIDF15.Rmd: Same as previous file, but I remove words that are too frequent in the corpus when creating the tf-idf matrix. The goal was to help the regularization algorithm in choosing the "best" model by removing words that are too frequent in the corpus to be of any use (which is a standard procedure in NLP anyway). Unfortunately, performance does not significantly change when structured data and texts are used in the model. If the text is the only covariate, performance decreases significantly. Therefore, I will not use this file to generate results table.
- Bert_300vs61.Rmd: comparison of performance of BERT models with embeddings passed through PCA. Using 300 dimensions vs (number of dimensions necessary to explain 80% of the variance)=61 is compared.

