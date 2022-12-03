# tfidf tests

################################
# Packages and import datasets #
################################
# global variables
home_directory <- "~/trulia_project"
setwd(home_directory)

# R packages
library(tidyverse)
library(magrittr) # use %$%
library(survival)
library(survminer)
library(lubridate) # %--%
library(tm) # tf-idf
library(FactoMineR) # PCA
library(factoextra)
library(feather)
library(riskRegression)
library(pec) # pec, crps
library(survAUC) # predErr

# my "packages"
source("generate_embeddings.R")
# functions to generate embeddings with tf-idf, transformer and fasttext
# function to put the embeddings in a dataframe

'%not in%' <- Negate('%in%')

#########################
# dataset of sold homes #
#########################

dfsold <- read.csv(file.path(home_directory, "data", "sold", "df_final.csv"))
str(dfsold)

df_sold <- dfsold %>% 
  select(-X) %>%
  mutate(
    # format baths, beds, lot_area, etc
    year_built = as.numeric(year_built)
    ,bath = as.numeric(bath)
    ,bed = as.numeric(bed)
    ,living_area = as.numeric(living_area)
    ,lot_area = as.numeric(lot_area)
    ,listed_date = lubridate::date(listed_date)
    ,sold_date = lubridate::date(sold_date)
    ,event = 1
  ) %>%
  distinct() # remove duplicate rows

# replace NAs with "No Info", which is already an existing category
df_sold <- df_sold %>%
  dplyr::mutate(
    # replace NA with "No Info"
    heating = ifelse(is.na(heating), "No Info", heating)
    ,cooling = ifelse(is.na(cooling), "No Info", cooling)
    ,parking = ifelse(is.na(parking), "No Info", parking)
    ,outdoor = ifelse(is.na(outdoor), "No Info", outdoor)
    ,pool = ifelse(is.na(pool), "No Info", pool)
    ,type = ifelse(type=="Unknown" | is.na(type), "No Info", type)
  )
summary(df_sold)

########################
# Exploratory analysis #
########################
#### Missing values
# There is nothing to do when listed_date or listed_price are NA, so I
# delete the observation. There is not much to do when sold_price is NA,
# so I also remove them.
# Very few NAs for restaurants, groceries, nightlife, year_built => removed
df_sold <- df_sold %>%
  drop_na(c("listed_price", "sold_price", "restaurants", "groceries", 
            "nightlife", "year_built"))
summary(df_sold)

# There is still a lot of NAs with bath, bed, living_area and lot_area.
# I will do nothing now but I probably will have to remove or impute the dataset

#### Analysis of prices

# some homes have a high listed_price but very low sold_price...
ggplot(df_sold) + 
  geom_point(aes(x=listed_price, y=sold_price, color=listed_price/sold_price))

df_sold %>%
  subset(listed_price/sold_price >= 20, select=-c(description, description_cased)) %>%
  arrange(sold_price) %>%
  print()

# I consider points with a high listed_price to sold_price ratio aberrant values
# I will therefore delete them. These points seem to be related to houses
# being on sale for a long time (listed_date < 2015)

df_sold2 <- df_sold %>%
  mutate(price_ratio = listed_price/sold_price) %>%
  subset(price_ratio < 20)

summary(df_sold2)

ggplot(df_sold2) + 
  geom_point(aes(x=listed_price, y=sold_price, color=price_ratio))

# other very small values of sold_price compared to listed_price seem to come 
# from the fact that the difference from listed_date and sold_date is high

df_sold2 %>%
  arrange(desc(price_ratio)) %>%
  subset(select=-c(description, description_cased)) %>%
  head(15)

# In some cases, the opposite happens : old homes are sold for a lot more than
# their initial price
df_sold2 %>%
  arrange(price_ratio) %>%
  subset(select=-c(description, description_cased)) %>%
  head(15)

# I think it is fair to say there is a "dynamic" that our variables might
# not be able to detect, hence the need for the description. This can be
# a motivation for including texts in a regression model trying to predict
# prices (even though we will instead try to predict time to sell).

# Some "homes" are actually whole apartment buildings (for investors).
# See below (df_censored) for more details.
# I will keep these observations in the dataset, hoping the embeddings
# will help the models will use this information for their predictions.


#################
# censored data #
#################
# Below, I do the same explanatory analysis as above, but for the censored dataset.

dfcensored <- read.csv(file.path(home_directory, "data", "censored", "df_final.csv"))
summary(dfcensored)

#### format variables
df_censored <- dfcensored %>%
  select(-X) %>%
  mutate(
    # format baths, beds, lot_area, etc
    year_built = as.numeric(year_built),
    bath = as.numeric(bath),
    bed = as.numeric(bed),
    living_area = as.numeric(living_area),
    lot_area = as.numeric(lot_area),
    listed_date = lubridate::date(listed_date),
    sold_date = lubridate::date(sold_date),
    event = 0
  ) %>%
  distinct() # remove duplicates
df_censored %>% str()
summary(df_censored)

df_censored <- df_censored %>%
  dplyr::mutate(
    # replace NA with "No Info"
    heating = ifelse(is.na(heating), "No Info", heating)
    ,cooling = ifelse(is.na(cooling), "No Info", cooling)
    ,parking = ifelse(is.na(parking), "No Info", parking)
    ,outdoor = ifelse(is.na(outdoor), "No Info", outdoor)
    ,pool = ifelse(is.na(pool), "No Info", pool)
    ,type = ifelse(type=="Unknown" | is.na(type), "No Info", type)
  ) %>%
  # drop observation if listed_price, listed_date or year_built ==NA
  # I also delete if NAs in other numeric variables for convenience...
  drop_na(c("listed_price", "listed_date", "year_built", 
            "groceries", "nightlife", "lot_area", "bed", "bath")) %>%
  subset(., subset=year_built > 1200) # there is one obs. with year_built==1195

summary(df_censored)

#### extreme values for listed_price, lot_area, bath and beds
ggplot(df_censored) +
  geom_point(aes(x=listed_price, y=lot_area))

ggplot(df_censored) +
  geom_point(aes(x=bed, y=lot_area))

ggplot(df_censored) +
  geom_point(aes(x=bath, y=lot_area))

ggplot(df_censored) +
  geom_point(aes(x=bed, y=bath)) # makes sense

df_censored %>%
  subset(subset=(type=="Residential Income")) %>%
  arrange(desc(bath)) %>%
  select(!description_cased) %>%
  head(10)

# according to the description, many of these are bricks, and not a single 
# house/apartment/condo/etc.
# It is written in the description, therefore we can hope a model
# with the description could be better at understanding that than a model
# without it.
# Therefore, I will not remove these observations.

df_censored$price_ratio <- NA


#### How big is the dataset if we delete NAs ?

# censored data : already deleted NAs
# sold homes : 
df_sold_nona <- df_sold2 %>%
  drop_na("bath", "bed", "lot_area")

df <- rbind(df_censored, df_sold_nona) %>% mutate(obs = 1:nrow(.))


###################################
# survival analysis : exploration #
###################################

###############################
# Other dataset modifications #
###############################
# Without removing NAs:
# df <- rbind(df_censored, df_sold2)

df$outdoor %>% table()
df$parking %>% table()
df$type %>% table(exclude=F)
df$subtype %>% table(exclude=F) # won't use it in models
# outdoor and parking :
# too many categories, I will group every modality with a parking/outdoor 
# to the value "Yes", otherwise "None" or "No Info"
# type and subtype :
# I won't use subtype, and I will try to regroup some categories of type.

#### listed_price and sold_price

# We will use listed_price/1000 to have more "grounded" coefficients.
# I will also divide lot_area by 1000
summary(df$listed_price)
plot(df$listed_price)

# Q: are there errors in dates?
# A: there are 3 ties
df %>%
  subset(subset=(listed_date >= sold_date)) %>%
  head()
# there are 3 homes that I scraped on september 2nd that were listed on september 2nd, which produces a tie in start and end time. I will remove them.

df <- df  %>%
  mutate(
    time = as.duration(listed_date %--% sold_date)/dmonths(1)
    ,parking = ifelse(parking %in% c("No Info", "None"), parking, "Yes")
    ,outdoor = ifelse(outdoor %in% c("No Info", "None"), outdoor, "Yes")
    ,listed_price = listed_price/100000
    ,lot_area = lot_area/1000
  ) %>%
  subset(subset=(listed_date < sold_date))

#### lot_area
# There are very small values of lot_area (sqrft)
#TODO : what to do with them?

#### Kaplan-Meier 
df$event %>% table() %>% prop.table()

# I don't think I can use counting process notation with Kaplan-Meier,
# so I use the time alive in the following graph
fit_km <- survfit(Surv(time, event) ~ 1, data = df)
ggsurvplot(
  fit = fit_km, 
  xlab = "Months", 
  ylab = "Overall survival probability",
  title = "Kaplan-Meier estimator of home survival")

#### Are there cases of complete separation?
df %$% table(type, event, exclude=F) # YES
df %$% table(cooling, event, exclude=F)
df %$% table(heating, event, exclude=F)
df %$% table(parking, event, exclude=F) # YES
df %$% table(outdoor, event, exclude=F) # yes
df %$% table(pool, event, exclude=F)

# This WILL cause problem when estimating the cox model. 
cox1 <- coxph(
  Surv(as.numeric(listed_date), as.numeric(sold_date), event) ~ listed_price + bed + bath + year_built + lot_area + heating + cooling + type + parking + outdoor + pool + restaurants + groceries+ nightlife
  , data=df
)
# To solve this issue, we can
# 1. combine categories
# 2. Remove variable
# 3. Collect more data
# I will start with option 1) and see what happens.

df <- df %>%
  mutate(
    type = case_when(
      type %in% c("Apartment", "Coop") ~ "apartment-coop"
      ,type %in% c("Residential Additional", "Residential Income") ~ "residential-other"
      ,type %in% c("Single Family Home", "Multi Family", "Townhouse") ~ "residential-other"
      ,TRUE ~ type # else condition
    )
    #,cooling=#TODO regler le coolingNoInfo NA
  )
df %$% table(type, event, exclude=F) %>% prop.table()*100 # still a problem. I will not use this variable in the model.
df %$% table(cooling, event, exclude=F) %>% prop.table()*100
df %$% table(heating, event, exclude=F) %>% prop.table()*100
df %$% table(parking, event, exclude=F) %>% prop.table()*100
df %$% table(outdoor, event, exclude=F) %>% prop.table()*100
df %$% table(pool, event, exclude=F) %>% prop.table()*100

#TODO
# might be problematic? The majority of observation with "No Info" weren't sold.

#####################
# Remove duplicates #
#####################

# there are a lot of duplicates which only differ from 1 or 2
# restaurants-groceries-nightlife.
duplicates <- df %>% 
  group_by(description, year_built, listed_date, sold_date, listed_price, 
           cooling, heating, parking, type,
           lot_area, living_area, bath, bed) %>% 
  filter(n()>1) %>%
  arrange(description)
# 25 > 6 : duplex therefore listed twice
# 6934 > 6925 : same home
# 7498 > 7476 : same home
# 6878 > 6860 : same home (two family)
# 6673 > 6662 : same home
# 6102 > 6098 : same home

# will have the index of the observations without the duplicates
remove_duplicates_index <- df[c("description", "year_built", "listed_date", "sold_date", "listed_price", 
                                "cooling", "heating", "parking", "type",
                                "lot_area", "living_area", "bath", "bed")] %>%
  duplicated()
df_nodup_1 <- df[!remove_duplicates_index,]

df3 <- df_nodup_1 %>%
  group_by(description) %>%
  filter(n()>1)

# after manual review, a few obs. are similar to other ones. I'm 99% sure it is 
# the same obs. but for some reason one has been listed as sold and the other
# one hasn't (even though I scraped sold homes before censored homes).
# The following is the house kept > house deleted
# 7363 > 7352 : same home with more info
# 6823 > 1123 : same home but 6823 is sold
# 6844 > 3082 : same home but 6844 is sold (two family victorian)
# 2237 > 1530 : same house with more info
# 3674 > 4873 : same home with more info
# 1630 > 590 : same infos except condo > residential-other and different resaurants & cie
df_final <- df_nodup_1 %>%
  subset(subset=(obs %not in% c(7352, 1123, 3082, 1530, 4873, 590)))
df <- df_final
df <- df %>% rename(num_bed=bed, num_bath=bath, I_outdoor=outdoor,
                    I_pool=pool, I_heating=heating, I_type=type, 
                    I_parking=parking, num_restaurants=restaurants,
                    num_groceries=groceries, num_nightlife=nightlife,
                    I_cooling=cooling, sale_time=time)

##############################
# homes on sale for too long #
##############################
# Somes homes have been on sale for 7+ years, which I find highly unprobable
df %>%
  subset(subset=!(lubridate::year(listed_date) <= 2014 & 
                    lubridate::year(sold_date) >= 2021)
         ,select=-c(description, description_cased)) %>%
  arrange(listed_date) %>%
  head()

# Houses on sale for >20 months or that took more than 20 months to sell
# are censored to 20 months

# update (december 20th) : instead of removing them, I censor their survival
# time at 20 months, as Thierry said I should do.
# Therefore, in subsequent files (3_variable_selection_and_performance2 and 
# corrupting_texts), remove these observations.

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

#############################################


my_stopwords <- c(tm::stopwords())
tfidf1 <- tf_idf_embedding(df$description_cased
                          ,removeNumbers = FALSE 
                          ,removePunctuation = TRUE 
                          ,stripWhitespace = TRUE
                          ,stemming = TRUE
                          ,tolower = TRUE
                          ,stopwords = my_stopwords
                          ,language = 'en'
                          ,wordLengths = c(3,Inf) # this is the default
                          ,SparseTermsPercent = 0.99 # remove most words that appear only in 1-SparseTermsPercent of texts
                          ,normalize_tfidf = TRUE
                          ,lower_IDF_quantile=NULL
                          ,low_document_frequency=1
                          )$tfidf %>%
  as.data.frame()

tfidf2 <- tf_idf_embedding(df$description_cased
                           ,removeNumbers = FALSE 
                           ,removePunctuation = TRUE 
                           ,stripWhitespace = TRUE
                           ,stemming = TRUE
                           ,tolower = TRUE
                           ,stopwords = my_stopwords
                           ,language = 'en'
                           ,wordLengths = c(3,Inf) # this is the default
                           ,SparseTermsPercent = 0.99 # remove most words that appear only in 1-SparseTermsPercent of texts
                           ,normalize_tfidf = TRUE
                           ,lower_IDF_quantile=0.15
                           ,low_document_frequency=1
                          )$tfidf %>%
  as.data.frame()


tfidf3 <- tf_idf_embedding(df$description_cased
                           ,removeNumbers = FALSE 
                           ,removePunctuation = TRUE 
                           ,stripWhitespace = TRUE
                           ,stemming = TRUE
                           ,tolower = TRUE
                           ,stopwords = my_stopwords
                           ,language = 'en'
                           ,wordLengths = c(3,Inf) # this is the default
                           ,SparseTermsPercent = 1 # remove most words that appear only in 1-SparseTermsPercent of texts
                           ,normalize_tfidf = TRUE
                           ,lower_IDF_quantile=NULL
                           ,low_document_frequency=75
)$tfidf %>%
  as.data.frame()


tfidf_df3 <- df_embeddings(df
                           ,text="description_cased"
                           ,checkpoint="tf-idf"
                              ,removeNumbers = FALSE 
                              ,removePunctuation = TRUE 
                              ,stripWhitespace = TRUE
                              ,stemming = TRUE
                              ,tolower = TRUE
                              ,stopwords = my_stopwords
                              ,language = 'en'
                              ,wordLengths = c(3,Inf) # this is the default
                              ,SparseTermsPercent = 0.99 # remove most words that appear only in 1-SparseTermsPercent of texts
                              ,normalize_tfidf = TRUE
                              ,lower_IDF_quantile=NULL
                              ,low_document_frequency=1
)
