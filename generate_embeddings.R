# R packages
library(reticulate)
library(tidyverse)
library(docstring) # documentation
library(tm) # tfidf and other text functions
library(pbapply) # progress bar
library(slam) # for IDF (WeightFunction)
library(R.utils) # gunzip, to extract .gz files

# Python modules
reticulate::use_python("~/anaconda3/envs/r_nlp2/bin/python", required=TRUE)
fasttext = reticulate::import('fasttext')
transformers = reticulate::import('transformers')
np = reticulate::import('numpy')

# setup environment and global variables
home_directory <- "~/trulia_project"
#setwd(home_directory)
'%not in%' <- Negate('%in%')

# I put this here so I can test the trf_embedding function. This will be
# in the main function called "df_embeddings"
# checkpoint = 'bert-base-uncased' # has been downloaded more times than bert-base-cased on hugginface
# #checkpoint = 'roberta-base'
# if (checkpoint == 'bert-base-uncased'){
#   tokenizer <- transformers$BertTokenizer$from_pretrained(checkpoint)
#   model <- transformers$BertModel$from_pretrained(checkpoint)
# } else if (checkpoint == 'camembert-base'){
#   tokenizer <- transformers$CamembertTokenizer$from_pretrained(checkpoint)
#   model <- transformers$CamembertModel$from_pretrained(checkpoint)
# } else if (checkpoint == 'roberta-base'){
#   tokenizer <- transformers$RobertaTokenizer$from_pretrained('roberta-base')
#   model = transformers$RobertaModel$from_pretrained('roberta-base')
# }

trf_count_length <- function(text, tokenizer){
  tokenized_sentence <- text %>%
    tokenizer$tokenize()
  return(length(tokenized_sentence))
}

trf_embedding <- function(text, tokenizer, model,
                          truncation = FALSE,
                          use_CLS = TRUE, 
                          layer=12, 
                          dimension=768){
  
  if (dimension != 768){
    stop("Using dimension != 768 hasn't been implemented yet.")
  }
  
  ########### if empty string, return a vector of 0 ########
  if (text==""){
    return(rep(0, times=dimension))
  }
  
  encoded_sentence <- text %>%
    tokenizer$tokenize() %>%
    tokenizer$encode(., truncation = truncation, return_tensors = 'pt')
  
  embeddings = model(encoded_sentence)
  
  if (use_CLS){
    to_return <- embeddings$last_hidden_state$squeeze()[0] # CLS token
  } else{
    stop("use_CLS is set to FALSE, but using layers is not implemented yet.")
  }
  # convert pytorch tensor to numpy array, which is then converted to a vector
  to_return <- to_return$detach()$numpy() %>% as.vector()
  #to_return <- as.vector(to_return)
  
  return(to_return) # vector of dim. 1x768
}
# Try embedding this long sentence with truncation=FALSE and TRUE. It is only
# possible with truncation=TRUE, and throws an error with truncation=FALSE.
# trf_embedding("This is a super long sentence. How long will it be? No one needs to know. Some weights of the model checkpoint at allenai/longformer-base-4096 were not used when initializing LongformerModel: ['lm_head.dense.bias', 'lm_head.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.layer_norm.weight'] This IS expected if you are initializing LongformerModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model). This is a super long sentence. How long will it be? No one needs to know. Some weights of the model checkpoint at allenai/longformer-base-4096 were not used when initializing LongformerModel: ['lm_head.dense.bias', 'lm_head.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.layer_norm.weight'] This IS expected if you are initializing LongformerModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model). This is a super long sentence. How long will it be? No one needs to know. Some weights of the model checkpoint at allenai/longformer-base-4096 were not used when initializing LongformerModel: ['lm_head.dense.bias', 'lm_head.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.layer_norm.weight'] This IS expected if you are initializing LongformerModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model). This is a super long sentence. How long will it be? No one needs to know. Some weights of the model checkpoint at allenai/longformer-base-4096 were not used when initializing LongformerModel: ['lm_head.dense.bias', 'lm_head.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.layer_norm.weight'] This IS expected if you are initializing LongformerModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).",
#               tokenizer, model,
#               truncation=TRUE)

# example 2:
# checkpoint = "bert-base-uncased"
# tokenizer <- transformers$BertTokenizer$from_pretrained(checkpoint)
# model <- transformers$BertModel$from_pretrained(checkpoint)
# trf_embedding("", model=model, tokenizer=tokenizer)

# This downloads the english fasttext model from
# https://fasttext.cc/docs/en/crawl-vectors.html
# if it hasn't been downloaded yet (if_exists='strict')
# if (!file.exists("fastText")){dir.create("fastText")}
# setwd("fastText")
# fasttext$util$download_model('en', if_exists='strict')
# # R.utils::gunzip("cc.en.300.bin.gz", remove=FALSE)
# setwd(home_directory)
ft_en <- fasttext$load_model('fastText/cc.en.300.bin')





fasttext_embedding <- function(text, model, dimension=300){
  #' Cree l'embedding d'un texte avec fasttext
  #'
  #' @param text string. Une phrase dans un string
  #' @param embed_model fasttext model (python), gere une langue seulement 
  #' @param dimension int. Nombre de dimensions de l'embedding
  #'
  #' @return matrix. embedding du texte
  if (dimension != 300){
    fasttext$util$reduce_model(model, as.integer(dimension))
  }
  embedding = model$get_sentence_vector(text)
  #return(t(as.matrix(embedding))) # pour avoir le meme format qu'avec spacy_trf_fr
  return(embedding)
}
# fasttext_embedding("test", model=ft_en)
# fasttext_embedding("100000$", model=ft_en)
#fasttext_embedding(df$description_cased[length(df)], model=ft_en)
#TODO fastText embeddings are made of smaller numbers than transformers,
# mainly Xe-02 instead of Xe-01. I don't know if this can impact results.


tf_idf_embedding <- function(corpus 
                             ,removeNumbers = FALSE 
                             ,removePunctuation = TRUE 
                             ,stripWhitespace = TRUE
                             ,stemming = TRUE
                             ,tolower = TRUE
                             ,stopwords = my_stopwords
                             ,language = 'en'
                             ,wordLengths = c(3,Inf) # this is the default
                             ,normalize_tfidf = TRUE
                             ,SparseTermsPercent = 0.9
                             ,lower_IDF_quantile=NULL
                             ,low_document_frequency=1
){
  #' Generate tf-idf embeddings. Words that are removed from the tfidf matrix (with lower_IDF_quantile or low_document_frequency) are removed AFTER tfidf is computed, so the values in the matrix do not change. If words are removed with sparseTermPercent, I think tfidf values change.
  #' 
  #' @param corpus A corpus of text, i.e. a vector with all the texts
  #' @param stopwords vector. Vector of stopwords (strings) to remove from text.
  #' @param wordLengths words will be considered in tf-idf only if length is in wordLenghts
  #' @param normalize_tfidf boolean (default=TRUE). Should vectors created with tfidf be normalized? Un-normalized vector will put a higher emphasis on longer texts (https://ethen8181.github.io/machine-learning/clustering_old/tf_idf/tf_idf.html)
  #' @param SparseTermsPercent Can be interpreted as : removes words that don't appear in (SparseTermsPercent*100)% of documents. Tf-idf is recomputed after removing unfrequent words (which might not be preferred). Smaller means less word in vector. Must be between (0,1]
  #' @param lower_IDF_quantile numeric (default=NULL). If not null, words appearing too much (with a low IDF value) will be removed if their IDF score is lower than the quantile of the distribution AFTER removing rare words
  #' @param low_document_frequency numeric (default=1). If not 1, words appearing in less than `low_document_frequency` documents will be removed from all documents. Tf-idf values are not recomputed.
  #' @param low_frequency numeric (default=1). Terms that appear in less documents than the lower bound `low_frequency` are discarded. Tf-idf values are recomputed after removing those words.
  #' @param high_frequency numeric (default=Inf). Terms that appear in more documents than the upper bound `high_frequency` are discarded. Tf-idf values are recomputed after removing those words.
  #' 
  #' @return (`tfidf`,`dtm`) Vector of (tfidf matrix, document-term matrix).
  
  if (SparseTermsPercent == 0){
    stop("SparseTermsPercent cannot be 0; cannot use 0 % of words in tf-idf.")
  }
  # might be necessary on windows:
  # corpus <- stringr::str_conv(corpus, "UTF-8")
  term_data <- corpus %>%
    VectorSource() %>%
    Corpus() %>%
    DocumentTermMatrix(., control = list(
      removeNumbers = removeNumbers
      ,removePunctuation = removePunctuation
      ,stripWhitespace = stripWhitespace
      ,stemming = stemming
      ,tolower = tolower
      ,stopwords = stopwords # or T for default stop words
      ,language = language
      ,wordLengths = wordLengths
      #,bounds = list(global = c(low_frequency, high_frequency)) # https://rdrr.io/rforge/tm/man/matrix.html
    )
    )
  
  #'removeSparseTerms()' only keep words present in x% of documents
  if (SparseTermsPercent != 1 | low_document_frequency!=1){
    term_data <- removeSparseTerms(term_data, SparseTermsPercent)
    warning("Make sure `SparseTermsPercent` and `low_document_frequency` are not both different than default values, because the two arguments have the same role.")
  }
  tf_idf <- as.matrix(weightTfIdf(term_data, normalize = normalize_tfidf))
  
  # words that are too frequent or not frequent enough will be removed from
  # the tfidf matrix, but tfidf will not be computed again (since we want
  # to do variable selection)
  
  # remove word that are too frequent (IDF lower than lower_IDF_quantile quantile)
  to_remove <- c()
  if (!is.null(lower_IDF_quantile)){
    idf <- extract_idf(term_data)
    words_overused <- idf %>% 
      subset(subset=(idf <= quantile(idf, prob=c(lower_IDF_quantile)))) %>%
      names()
    to_remove <- c(to_remove, words_overused)
  }
  
  # Remove words not frequent enough (similar to SparseTermData)
  # instead of going through the following condition, one could use
  # control(bounds=c(low_frequency, high_frequency). But that would mean
  # recomputing tfidf after removing words
  if (low_document_frequency != 1){
    isDTM <- inherits(term_data, "TermDocumentMatrix")
    if (isDTM) term_data <- t(term_data)
    
    # term_data_01 is a TRUE or FALSE matrix. 
    # (a_ij) indicating if word j is in document i
    term_data_01 <- (term_data > 0) %>% as.matrix()
    class(term_data_01) <- "numeric"
    document_frequency <- col_sums(term_data_01)
    low_freq_words <- document_frequency %>%
      subset(subset=document_frequency <= low_document_frequency) %>%
      names()
    
    to_remove <- c(to_remove, low_freq_words)
  }
  
  # After computing tfidf, we remove words that are too frequent and
  # those that are not frequent enough :
  tf_idf <- tf_idf[, colnames(tf_idf) %not in% to_remove]
  
  return(list(tfidf = tf_idf
              ,dtm = term_data))
}

# copy-pasted from tm documentation: https://github.com/cran/tm/blob/master/R/weight.R#L17
# see function weightTfIdf
extract_idf <- 
  WeightFunction(function(m, normalize = TRUE) {
    #' @param m document-term matrix : matrix with frequency of each words in the document. This is the tf term.
    #' @param normalize default TRUE : normalize vector in idf to remove bias towards long documents (since a term has more change to appear in a longer document).
    
    isDTM <- inherits(m, "DocumentTermMatrix")
    if (isDTM) m <- t(m)
    if (normalize) {
      # col_sums from slam package: https://stackoverflow.com/questions/15055584/cant-find-documentation-for-r-function-row-sums-and-col-sums
      cs <- slam::col_sums(m) 
      if (any(cs == 0))
        warning("empty document(s): ",
                paste(Docs(m)[cs == 0], collapse = " "))
      names(cs) <- seq_len(nDocs(m))
      m$v <- m$v / cs[m$j]
    }
    rs <- slam::row_sums(m > 0)
    if (any(rs == 0))
      warning("unreferenced term(s): ",
              paste(Terms(m)[rs == 0], collapse = " "))
    # lnrs is the IDF term
    # some article recommand to use add 1 to the denominator to avoid division by zero
    lnrs <- log2(nDocs(m) / rs)
    lnrs[!is.finite(lnrs)] <- 0 # this is used instead of using +1 in the denom
    #print(paste("no division by 0:", all(is.finite(lnrs))))
    
    # return :
    return(lnrs)
  }, "term frequency - inverse document frequency", "tf-idf")

ghist_idf <- function(data, bins=30, title="Histogram of Inverse Document Frequency"){
  ##   Fonction permettant d'effectuer des ggplot déjà formatés
  ##
  ##   Args:
  ##     data: Un jeu de données (data.frame)
  ##     variable: Variable étudiée
  ##     title: Un titre (character)
  ##
  ##   Returns:
  ##      Retourne un graphique (ggplot)
  
  df <- as.data.frame(data)
  
  gplot <- ggplot(as.data.frame(df), aes(x=data)) + 
    geom_histogram(bins=bins, color="white") +
    scale_x_continuous(breaks = round(seq(min(df[,"data"]), max(df[,"data"]), by = 1),1)) +
    stat_bin(aes(y=..count.., label=..count..), geom="text", vjust=-.5, bins=bins) +
    theme_bw() + 
    labs(x="IDF")
  
  if (title!=""){
    gplot <- gplot+ggtitle(label=title)
  }
  return(gplot)
}





df_embeddings <- function(df, text, checkpoint="bert-base-uncased", ..., 
                          dimension=NULL, 
                          dimension_reduction=NULL){
  #' Ajoute l'embedding des textes au dataframe.
  #'
  #' @param df data.frame. Dataset qui contient les donnees structurees et une colonne avec les textes. Does not work with tibble.
  #' @param text string. Nom de la colonne qui contient les textes
  #' @param checkpoint string. Transformers : bert-base-uncased (en seulement), camembert (fr seulement), xlm-enfr (fr et en). fastText : fasttext (fr seulement)
  #' @param ... Arguments qui seront passes aux fonctions qui font les embeddings.
  #' @param dimension integer. Nombre de dimensions des vecteurs
  #' @param dimension_reduction string. Si dimension != NULL, methode de reduction de la dimensionnalite
  #'
  #' @return data.frame. df initial auquel a ete ajoute l'embedding des textes
  
  #TODO: df must be a data.frame object, it does not work with, e.g. a tibble
  # when using checkpoint="tf-idf"
  
  # Gestion de la dimension
  if (is.null(dimension) & checkpoint!="fasttext"){
    dimension = 768 # dimension par defaut avec BERT-like models
  } else if (is.null(dimension) & checkpoint=="fasttext"){
    dimension = 300 # dimension par defaut avec fastText
  }
  
  # Gestion du modele pour les embeddings
  if (checkpoint %in% c('bert-base-uncased', 'roberta-base')){
    
    # it would be easier to use AutoTokenizer, but I've had problems with
    # it in the past... so I use this condition to import the appropriate model.
    if (checkpoint == 'bert-base-uncased'){
      tokenizer <- transformers$BertTokenizer$from_pretrained(checkpoint)
      model <- transformers$BertModel$from_pretrained(checkpoint)
    } else if (checkpoint == 'camembert-base'){
      tokenizer <- transformers$CamembertTokenizer$from_pretrained(checkpoint)
      model <- transformers$CamembertModel$from_pretrained(checkpoint)
    } else if (checkpoint == 'roberta-base'){
      tokenizer <- transformers$RobertaTokenizer$from_pretrained('roberta-base')
      model <- transformers$RobertaModel$from_pretrained('roberta-base')
    }
    
    # TODO FOR NEXT PROJECT :
    # model$eval() # the model shouldn't be in "train" mode, should be faster because gradient isn't kept
    # TODO : generate embeddings on GPU? : https://stackoverflow.com/q/62385092
    # https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models -> sentence bert could be faster?
    # they support fr and fr_ca with multilingual models
    # model$to(device) ???

    # sapply retourne un vecteur, plus rapide que lapply
    # pbsapply : sapply with progress bar
    embeddings <- pbsapply(X=df[, text]
                        ,FUN=trf_embedding
                        ,model=model 
                        ,tokenizer=tokenizer
                        ,...
                        ) %>%
                        t() # matrix of dim. nrow(df)x768
    #print(embeddings[1])
  
  } else if (checkpoint=="fasttext"){

    embeddings <- sapply(X=df[, text]
                          ,FUN=fasttext_embedding
                          ,model=ft_en
                          ,dimension=dimension
                          ) %>%
                          t()

  } else if (checkpoint=="tf-idf"){

    # this is a vector of (tfidf, document-term matrix)
    # here, we are interested in tfidf values
    embeddings <- tf_idf_embedding(corpus = df[, text] 
                                   ,...
                                   )$tfidf

  } else{
    stop("Checkpoint must be bert-base-uncased, tf-idf or fasttext.")
  }
  
  # retourne un df avec les dimensions pour chaque texte
  embeddings_df = data.frame(embeddings)
  
  #print(embeddings_df)
  if (checkpoint == "fasttext"){
    colnames(embeddings_df) <- paste0("ft", 1:length(embeddings_df))
    rownames(embeddings_df) <- c() # remove rownames
  } else if (checkpoint != "tf-idf"){ # transformer model
    colnames(embeddings_df) <- paste0("V", 1:length(embeddings_df))
  }

  #TODO
  # PCA/t-SNE/UMAP pour les modèles BERT ici
  # t-SNE/UMAP pour fasttext ici?
  
  # ajout des embeddings au df initial
  #return(t(embeddings))
  return(cbind(df, embeddings_df))
}
# a <- df_embeddings(df[1:3, ], "description_cased", checkpoint="tf-idf", SparseTermsPercent=0.9)
# a <- df_embeddings(df[1:2, ], "description_cased", checkpoint="bert-base-uncased")
# a <- df_embeddings(df[1:3, ], "description_cased", checkpoint="fasttext")

#with pbsapply
# benchmark({df_embeddings(df[1:2, ], "description_cased", checkpoint="bert-base-uncased")},
#           replications=10)
# replications   elapsed relative user.self  sys.self user.child sys.child
# 1           10 165.925        1    267.42    0.623          0         0

#with sapply
# benchmark({df_embeddings(df[1:2, ], "description_cased", checkpoint="bert-base-uncased")},
#           replications=10)
# replications   elapsed  relative user.self sys.self user.child sys.child
# 1           10 165.593        1   265.921    0.521          0         0
#TODO 
# faire une fonction pour utiliser le package text?