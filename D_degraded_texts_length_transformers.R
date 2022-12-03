# Objective : Verify the length difference in the tokenized sentences between 
# original texts and degraded texts

#########
# Setup #
#########

home_directory <- "~/trulia_project"
setwd(home_directory)

set.seed(1)

# R packages
library(tidyverse); library(magrittr); library(survival); library(survminer); library(lubridate); library(tm); library(FactoMineR); library(factoextra); library(feather); library(riskRegression); library(pec); library(survAUC); library(glmnet); library(plotmo); library(quanteda); library(R.utils); library(patchwork)
# magrittr : use %$%
# lubridate : %--%
# tm : tf-idf
# FactoMineR : PCA
# pec : pec, crps
# survAUC : predErr
# plotmo : plot_glmnet

# R.utils : to insert elements to a vector. If not available, try to use "append", but insert is more flexible

# Python setup
reticulate::use_python("~/anaconda3/envs/r_nlp2/bin/python", required=TRUE)
transformers = reticulate::import('transformers')

# Others
'%not in%' <- Negate('%in%')

####################################
# create/import necessary datasets #
####################################

df_bbu <- read_feather("embeddings/bert_base_uncased.feather")
df <- df_bbu %>%
  select(!V1:V768)
# remove everything that is related to bert embeddings

# Houses on sale for >20 months or that took more than 20 months to sell
# are censored to 20 months
df <- df %>%
  mutate(
    to_modify = case_when(sale_time > 20 ~ 1
                          ,TRUE ~ 0 # else
    )
    , event = case_when(to_modify == 1 ~ 0 # censoring events flagged as to_modify
                        ,TRUE ~ event)
    , sale_time = case_when(to_modify ==1 ~ 20 # sale_time forced to 20 months
                            ,TRUE ~ sale_time)
  )
# before censoring events that took too long, there were 1567 events. After
# censoring at 20 months, there where 1537 events (20% of the dataset).

# REMARK : 
# For the function extract_nonzero_coeff to work, variables treated as
# factors must be transformed as factor here
df <- df %>%
  mutate(
    # format baths, beds, lot_area, etc
    year_built = as.numeric(year_built)
    ,num_bath = as.numeric(num_bath)
    ,num_bed = as.numeric(num_bed)
    ,living_area = as.numeric(living_area)
    ,lot_area = as.numeric(lot_area)
    ,listed_date = lubridate::date(listed_date)
    ,sold_date = lubridate::date(sold_date)
    ,I_heating = as.factor(I_heating)
    ,I_cooling = as.factor(I_cooling)
    ,I_parking = as.factor(I_parking)
    ,I_outdoor = as.factor(I_outdoor)
    ,I_pool = as.factor(I_pool)
    ,I_type = as.factor(I_type)
    ,event = as.numeric(event)
  ) %>%
  # need to have no spaces in categories for regex
  mutate(
    I_heating = recode(I_heating, "No Info" = "No_Info")
    , I_cooling = recode(I_cooling, "No Info" = "No_Info")
    , I_parking = recode(I_parking, "No Info" = "No_Info")
    , I_outdoor = recode(I_outdoor, "No Info" = "No_Info") 
    , I_pool = recode(I_pool, "No Info" = "No_Info")
    , I_type = recode(I_type, "No Info" = "No_Info")
  )

#### corrupted_text_0typos
source("generate_embeddings.R")
corrupted_text_0typos <- read_feather("./corrupted_text/0typos.feather")

checkpoint <- "bert-base-uncased"
tokenizer <- transformers$BertTokenizer$from_pretrained(checkpoint)
tokenized_length_original <- sapply(df$description_cased, trf_count_length, tokenizer, USE.NAMES=FALSE)
tokenized_length_degraded <- sapply(corrupted_text_0typos$corrupted_text_0typos, trf_count_length, tokenizer, USE.NAMES=FALSE)
diff <- tokenized_length_orginal - tokenized_length_degraded

token_counts <- data.frame(original=tokenized_length_original,
                           degraded=tokenized_length_degraded,
                           diff=diff)

ggplot(token_counts) +
  geom_histogram(aes(x=original, fill="red", alpha=0.2)) +
  geom_histogram(aes(x=degraded, fill="blue", alpha=0.2)) + 
  geom_vline(xintercept=512, color="red") +
  labs(title = "Token count per sentence", x = "Number of token", y = "Count", color = "Legend Title\n") +
  scale_fill_manual(name="Text type",labels=c("original","degraded"),values=c("red","blue"))

(tokenized_length_degraded > 512) %>% sum()
(tokenized_length_original > 512) %>% sum()

(tokenized_length_degraded <= 512 & tokenized_length_original > 512) %>% sum()
# for 198 out of 7543 texts (2.6%), the transformer with degraded text has
# more info 

tokenized_length_degraded[tokenized_length_original > 512] %>% qplot()
