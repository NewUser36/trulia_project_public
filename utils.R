# train-test-split
train_valid_test <- function(df, id_column, p_train, p_valid, p_test=1-p_train-p_valid, type="stratified", random_seed=1){
  #' Splits data in three parts (train, validation, test)
  #'
  #' @param df data.frame. The dataset 
  #' @param id_column string. Variable on which we wish to split
  #' @param p_train numeric. proportion of ids in training set, or approximative proportion of data for training
  #' @param p_valid numeric. See p_train
  #' @param type str. Façon dont les données sont séparées. "stratified" (default), "grouped" pour clustered or panel data
  #' @param random_seed int. Seed used to randomly split data. Default is 1.
  #'
  #' @return \item{train}{data.frame. training set} \item{valid}{data.frame. validation set} \item{test}{data.frame. testing set}
  
  if(p_train + p_valid + p_test != 1){
    stop("sum of probabilities must be 1")
  }
  df <- as.data.frame(df)
  
  ids_split <- splitTools::partition(
    y = df[, id_column]
    # y=dplyr::pull(df[, id_column]) # pull nécessaire car pbc est un tibble
    ,p = c(train = p_train, valid = p_valid, test = p_test)
    ,type = type
    ,seed = random_seed
  )
  train <- df[ids_split$train, ]
  valid <- df[ids_split$valid, ]
  test <- df[ids_split$test, ]
  
  return(list(train=list(df=train, prop=nrow(train)/nrow(df)),
              valid=list(df=valid, prop=nrow(valid)/nrow(df)),
              test=list(df=test, prop=nrow(test)/nrow(df))
  )
  )
}

outersect <- function(a,b){
  #' Inverse of intersect(). Returns what is not common between two sets a and b.
  unique(c(setdiff(a,b), setdiff(b,a)))
} 

extract_categories <- function(variable, df){
  categories <- df[, variable] %>% unique()
  categories
}

extract_var_categories <- function(df, variables_names){
  if (is_tibble(df)){
    df <- as.data.frame(df)
  }
  
  if (variables_names==""){ # if no variable in the model
    return(NULL)
  }
  
  # extract variables that are factors or characters
  factors <- c()
  for (variable in variables_names){
    if (df[, variable] %>% class() %in% c("character", "factor"))
      factors <- c(factors, variable)
  }
  
  # extract categories of character/factor variables
  liste <- sapply(factors, extract_categories, df, USE.NAMES=TRUE)
  return(liste)
}

merge_factors <- function(factor, factors_to_merge, merge_to){
  #' This function is used to merge factors of a categorical variable together. 
  #' Levels in `factors_to_merge` will become `merge_to`. 
  #' 
  #' @param factor factor. Factor to modify
  #' @param factors_to_merge vector of string. Levels in factor to be merged with `merge_to`
  #' @param merge_to str.
  #' 
  #' @return factor. Factor with modified levels
  pattern <- paste(factors_to_merge, collapse="|")
  
  new_levels <- str_replace_all(levels(factor), 
                                pattern=pattern, 
                                replacement=merge_to
  )
  
  levels(factor) <- new_levels
  return(factor)
}
# extract_var_categories(df, variables_names=c("listed_price", "num_bed", "num_bath", "year_built", "lot_area", "I_heating", "I_parking", "I_outdoor", "I_pool", "num_restaurants", "num_groceries"))

extract_zero_nonzero_coeff <- function(cvglmnet, lambda_value){
  #' Extract coefficients of a glmnet model that became 0 due to regularization.
  #'
  #' @param cvglmnet cv.glmnet object.
  #' @param lambda_value str. "lambda.1se" or "lambda.min"
  #'
  #' @return list. zeros : vector of variable names that became 0. non_zeros : vector of variable names that stayed significative (non zero) according to the regularization algorithm.
  
  tmp_coeffs <- coef(cvglmnet, s = lambda_value)
  non_zero_position <- tmp_coeffs@i
  
  # count starts at 0   #-1 : count starts at 0 but length starts at 1                            
  zeros_position <- seq(0, length(tmp_coeffs)-1) %>% outersect(., non_zero_position) + 1 # +1 because count starts at 0 but R starts listing at 1
  zeros_variables <- tmp_coeffs@Dimnames[[1]][zeros_position]
  ## zeros_variables : I_heatingNo, I_heatingYes, I_heating_No_Info, etc.
  ## depending on what the model selected
  
  non_zeros_variables <- tmp_coeffs@Dimnames[[1]][tmp_coeffs@i+1] # +1 because count starts at 0
  
  return(list(zeros = zeros_variables,
              non_zeros = non_zeros_variables))
}

create_new_df_regularized <- function(cvglmnet, 
                                      lambda_value="lambda.1se", 
                                      variables_names, 
                                      embedding_names="", 
                                      dataset){
  #' Creates a new dataset based on the important variable determined by a cv.glmnet object. Factor levels that are deleted by the algorithm are merged to the reference level.
  #'
  #' @param cvglmnet cv.glmnet object
  #' @param lambda_value numeric. default="lambda.1se". Value of lambda to use in coef(cv.glmnet, s=lambda)
  #' @param variable_names str. Vector of strings with all the variables used in the Cox model, except the variables refering to the embeddings.
  #' @param embedding_names str (optional). Variables refering to the text embeddings.
  #' @param dataset data.frame or tibble. Train test used to train the model used in cv.glmnet
  #' @param surv_call str. Survival formula of the cox model in string format. e.g. "Surv(time1, time2, event)"
  #'
  #' @return tmp_dataset. `tmp_dataset`: new dataset made from the original dataset, but with factor variable merged with reference value when they were not selected by the algorithm.
  
}

generate_newcox_glmnet <- function(cvglmnet, 
                                   lambda_value="lambda.1se", 
                                   variables_names, 
                                   embedding_names="", 
                                   dataset,
                                   dataset_test,
                                   surv_call="Surv(sale_time, event)"
){
  #' Generate a new cox model based on the important variable determined by a cv.glmnet object. Factor levels that are deleted by the algorithm are merged to the reference level. This function assumes that the train and test sets have the same factors for every categorical variables (which should be true if both dataset are big enough).
  #'
  #' @param cvglmnet cv.glmnet object
  #' @param lambda_value numeric. default="lambda.1se". Value of lambda to use in coef(cv.glmnet, s=lambda)
  #' @param variable_names str. Vector of strings with all the variables used in the Cox model, except the variables refering to the embeddings.
  #' @param embedding_names str (optional). Variables refering to the text embeddings.
  #' @param dataset data.frame or tibble. Dataset used to train the model used in cv.glmnet
  #' @param dataset_test data.frame or tibble. Dataset used to assess performance of cox model. This function simply applies the same transformations that are applied to `dataset`.
  #' @param surv_call str. Survival formula of the cox model in string format. e.g. "Surv(time1, time2, event)"
  #'
  #' @return cox, train_df, test_df `cox`: cox model evaluated on tmp_dataset. `train_df`: new dataset made from the original dataset, but with factor variable merged with reference value when they were not selected by the algorithm. `test_df`: same as `train_df`, but for the test/validation set.
  
  # some dataset manipulation can't work if dataset is a tibble, so I convert it
  # here. I will later transform this temporary df to merge factors if necessary 
  tmp_dataset <- as.data.frame(dataset)
  tmp_dataset_test <- as.data.frame(dataset_test) # test/validation set
  
  ########################################################
  # Create new dataset without factors removed by glmnet #
  ########################################################
  
  zeros_non_zeros_variables <- extract_zero_nonzero_coeff(cvglmnet, lambda_value)
  
  # extract variable names with coefficient equal to 0 in glmnet
  zeros_variables <- zeros_non_zeros_variables$zeros
  ## e.g. : I_heatingNo, I_heatingYes, I_heating_No_Info, etc.
  ## depending on what the model selected
  
  # extract categories of factor/character variables
  list_factors <- extract_var_categories(tmp_dataset, variables_names)
  factor_variables <- names(list_factors)
  ## factor_variables = I_heating, I_parking, I_pool, I_outdoor
  
  for (factor_variable in factor_variables){
    pattern = paste0("(", factor_variable, ")", "([[:alpha:]]+[_]?[[:alpha:]]+)")
    
    # extract VariableFactor from the list of variable with zero coefficient (zeros_variables)
    zeros_variableFactor <- regmatches(zeros_variables, regexpr(pattern, zeros_variables))
    ## e.g. c("I_heatingNo", "I_heatingYes")
    
    # zeros_variableFactor could be empty if no level is removed from the factor
    # so we have to condition before trying to merge levels
    if (!identical(character(0), zeros_variableFactor)){
      
      # extract factor from the VariableFactor vector
      factors_to_merge_with_reference <- gsub(x=zeros_variableFactor
                                              ,pattern=factor_variable
                                              ,replacement=""
      )
      ## e.g. c("No", "Yes")
      
      # merge levels deleted by the algorithm in the reference level
      reference <- levels(tmp_dataset[, factor_variable])[1] # first level is always the reference
      
      # train set
      tmp_dataset[, factor_variable] <- merge_factors(tmp_dataset[, factor_variable]
                                                      ,factors_to_merge=factors_to_merge_with_reference
                                                      ,merge_to=reference
      )
      # test set
      tmp_dataset_test[, factor_variable] <- merge_factors(tmp_dataset_test[, factor_variable]
                                                           ,factors_to_merge=factors_to_merge_with_reference
                                                           ,merge_to=reference
      )
    }
    
    # if there is only one level in factor, the variable contains no information 
    # anymore and should be removed from coxph formula to make sure there is no
    # error with coxph.
    to_remove <- c()
    if (tmp_dataset[, factor_variable] %>% levels() %>% unique() %>% length() ==1){
      to_remove <- c(to_remove, factor_variable)
    }
  }
  # if to_remove does not exist, it is because variable_names is empty (cox model
  # is only made of embedding names/variables related to text). To prevent
  # an error, we create an empty string vector.
  if (!exists("to_remove")){
    to_remove <- c("")
  }
  ######################################################
  # create new cox model with variables kept by glmnet #
  ######################################################
  
  ## extract non-zero variables from glmnet
  non_zero_variablesFactor <- zeros_non_zeros_variables$non_zeros
  ## e.g. num_bed, I_heatingNo, I_heatingYes, listed_price,...
  
  ## extract names of the variables without the added factor
  pattern1 = paste0(embedding_names, collapse = "$|") %>% # exact match on embedding dimensions
    paste0(., "$|")
  pattern2 = paste0(variables_names, collapse="|", sep="") # partial match on variable names (to exclude the factor : I_heatingNo becomes I_heating)
  pattern_12 = paste0(pattern1, pattern2)
  
  non_zero_variables <- stringr::str_extract(string=non_zero_variablesFactor, pattern=pattern_12) %>%
    unique()
  
  # remove factors with only one level
  #TODO HAVEN'T TESTED THIS, SO MIGHT CAUSE AN ERROR
  non_zero_variables <- gsub(non_zero_variables, 
                             pattern=paste(to_remove, collapse="|"), 
                             replacement=""
  )
  
  new_cox_formula <- paste0(surv_call, "~", paste0(non_zero_variables, collapse="+")) %>%
    as.formula()
  
  # running cox model with new dataset and selected variables
  cox_with_new_dataset <- coxph(
    new_cox_formula
    ,data=tmp_dataset
  )
  
  ################################
  # performance of new cox model #
  ################################
  #TODO
  #create a function for this that does cross validation
  
  return(list(cox = cox_with_new_dataset
              ,train_df = tmp_dataset
              ,test_df=tmp_dataset_test
  )
  )
}

# Unfortunately, Harrell's c [concordance index or C-index] is also less sensitive than the above statistics, so you may not want to choose between models based on it if the difference between them is small; it's more useful as an interpretable index of general performance than a way to compare different models. (source : https://stats.stackexchange.com/a/133877)
performance_measures <- function(model, 
                                 olddata, 
                                 newdata, 
                                 times=NULL,
                                 surv_call="Surv(sale_time, event)"
                                 ,boot_sample=150){
  #' Returns several performance measures available for cox models (coxph). As far as I know, these scores are only available for cox models of the form Surv(time, event), that is, without left truncation or interval censoring.
  #'
  #' @param model coxph model to evaluate
  #' @param olddata training data
  #' @param newdata test data
  #' @param times (optional) times is a necessary parameter for Brier score and AUC. It is the time at which the brier score/AUC is computed. If no value is given, times will be every time an event occurs in the test set.
  #' @param surv_call Expression of the survival function for the cox model of the form Surv(time, event).
  #' @param boot_sample Number of bootstrap samples to use
  #'
  #' @return scores. Scores include Brier score (at median time and integrated for all time), concordance index (standard concordance index, Uno's C index and bootstrapped C index), AUC at median time, and integrated AUC. Three methods for computing AUC are used (Chambless and Diao, Stat Med 2006;25:3474-3486., Song and Zhou (Biometrics 2011;67:906-16) and Uno et al. (http://biostats.bepress.com/cgi/viewcontent.cgi?article=1041&context=harvardbiostat). 
  
  if (is.null(times)){
    #TODO this is hardcoded...
    times <- newdata[newdata$event==1,]$sale_time %>% sort() %>% unique()
  }
  
  # I could also use model$linear.predictors
  lp_train <- predict(model, type="lp") # lp stands for linear predictor
  lp_test <- predict(model, newdata=newdata, type="lp")
  
  # I keep "~" in the pattern to make sure I get the entire call, wether it is
  # Surv(event, time) or Surv(start, stop, event)
  # pattern <- "Surv\\((.*?)\\) ~"
  # model_call <- as.character(model$call)
  # call = str_extract(model_call, pattern)[2] %>%
  #   substr(., start=1, stop=nchar(.)-2) # to remove " ~"
  call=surv_call
  
  # needs to be parsed before putting it in with(oldata, formula)
  Surv_formula <- parse(text=call)
  
  Surv_train <- with(olddata, eval(Surv_formula))
  Surv_test <- with(newdata, eval(Surv_formula))
  
  ###########
  # Uno's C #
  ###########
  # values can be in (0.5, 1)
  C_uno <- survAUC::UnoC(Surv_train, Surv_test, lp_test)
  
  ###########
  # C index #
  ###########
  # Hmisc::rcorrcens
  # https://rpubs.com/kaz_yos/survival-auc : A larger marker value is considered to be associated with a longer survival by this function. Thus, the linear predictor (the higher, the worse) needs to be negated
  # they do x=-I*lpnew, but instead I do 1-rcorr.cens
  C_index <- (1-Hmisc::rcorr.cens(x=lp_test, S=Surv_test)[1])
  #C_index <- c_index(model, newdata)
  
  ##########################
  # integrated brier score #
  ##########################
  brier <- survAUC::predErr(Surv_train, Surv_test, lp_train, lp_test, times, 
                            type = "brier", int.type = "weighted")
  # weighted because we want to weight brier score by the inverse censoring 
  # distribution estimated with Kaplan-Meier. Usually the "brier score" is 
  # defined as the weighted brier score, so this is what I do here.
  # brier$ierror is the integrated brier score
  
  # I decided not to use Brier and integrated brier score because the 
  # values should be between 0.25 and 0.5, but I get 6.8e+08
  
  ##############################
  # brier score at median time #
  ##############################
  #TODO read this to make sure this code is ok
  # https://www.jesseislam.com/post/brier-score/
  
  # do we compute the median time on train (olddata) or test (newdata) set?
  # do we compute the median time for all data (event==1 | event==0)
  # or only events (event==1)?
  median_time <- olddata$sale_time %>% median()
  brier_med <- survAUC::predErr(Surv_train, Surv_test, lp_train, lp_test, median_time, 
                                type = "brier", int.type = "weighted")
  # I decided not to use Brier and integrated brier score because the 
  # values should be between 0.25 and 0.5, but I get 6.8e+08
  
  ##########################################
  # AUC(t) at chosen time (median time)    #
  # https://rpubs.com/kaz_yos/survival-auc #
  ##########################################
  # model_formula <- model$formula # Surv(time, event ~ x1+x2+...)
  # model_test <- coxph(model_formula, data = newdata)
  
  # Cumulative case/dynamic control AUC by Chambless and Diao (Stat Med 2006;25:3474-3486.)
  auc_cd_median <- AUC.cd(Surv.rsp = Surv_train,
                          Surv.rsp.new = Surv_test,
                          lp = lp_train,
                          lpnew = lp_test,
                          times = median_time)
  
  # Incident case or Cumulative case/dynamic control AUC by Song and Zhou (Biometrics 2011;67:906-16)
  # THIS USES INCIDENT SENSITIVITY INSTEAD OF CUMULATIVE because I didn't know 
  # the default was "incident"
  auc_sh_median <- AUC.sh(Surv.rsp = Surv_train,
                          Surv.rsp.new = Surv_test,
                          lp = lp_train,
                          lpnew = lp_test,
                          times = median_time,
                          type="incident")
  
  # Cumulative case/dynamic control AUC by Uno et al.
  # (http://biostats.bepress.com/cgi/viewcontent.cgi?article=1041&context=harvardbiostat)
  auc_uno_median <- AUC.uno(Surv.rsp = Surv_train,
                            Surv.rsp.new = Surv_test,
                            #lp = lp_train,
                            lpnew = lp_test,
                            times = median_time)
  
  ##################
  # Integrated AUC #
  ##################
  
  # 'times' will be the event time in the test dataset
  # Cumulative case/dynamic control AUC by Chambless and Diao (Stat Med 2006;25:3474-3486.)
  auc_cd <- AUC.cd(Surv.rsp = Surv_train,
                   Surv.rsp.new = Surv_test,
                   lp = lp_train,
                   lpnew = lp_test,
                   times = times)
  
  # Incident case or Cumulative case/dynamic control AUC by Song and Zhou (Biometrics 2011;67:906-16)
  # THIS USES INCIDENT SENSITIVITY INSTEAD OF CUMULATIVE because I didn't know 
  # the default was "incident"
  auc_sh <- AUC.sh(Surv.rsp = Surv_train,
                   Surv.rsp.new = Surv_test,
                   lp = lp_train,
                   lpnew = lp_test,
                   times = times,
                   type="incident")
  
  # Cumulative case/dynamic control AUC by Uno et al.
  # (http://biostats.bepress.com/cgi/viewcontent.cgi?article=1041&context=harvardbiostat)
  auc_uno <- AUC.uno(Surv.rsp = Surv_train,
                     Surv.rsp.new = Surv_test,
                     #lp = lp_train,
                     lpnew = lp_test,
                     times = times)
  
  #######################
  # Scores by bootstrap #
  #######################
  # Gives a bunch of statistics on the function
  # evaluated by bootstrap. I am interested in the concordance index
  # obtained by boostrap.
  # https://stats.stackexchange.com/questions/17480/how-to-do-roc-analysis-in-r-with-a-cox-model
  
  # This might generate new convergence warnings/problems because there is less data variability in the data and we are trying to fit models with a lot of variables.
  
  # cph_model <- rms::cph(model$formula, x=T, y=T, data=olddata)
  # v <- rms::validate(cph_model, dxy=TRUE, B=boot_sample, method="boot")
  # Dxy_boot = v[rownames(v)=="Dxy", colnames(v)=="index.corrected"]
  # c_boot <- Dxy_boot/2+0.5
  
  # Update December 23rd : I remove it because it's too long to compute
  # for more complex models and the value is very simial to C_index on 
  # test data (as expected)
  
  return(list(
    brier_med=brier_med$error
    ,integrated_brier_score=brier$ierror
    ,C_uno = C_uno
    ,C_index = C_index
    #,C_index_bootstrap = c_boot
    ,AUC_median = list(cd=auc_cd_median$auc
                       ,sh=auc_sh_median$auc
                       ,uno=auc_uno_median$auc
    )
    ,IAUC = list(cd=auc_cd$iauc
                 ,sh=auc_sh$iauc
                 ,uno=auc_uno$iauc
    )
  )
  )
}
# performance_measures(full, train, test) 
# sh takes more time to compute
# uno takes a little while but not as long as sh
# hc gives very large value (does not make sense)

# I think the brier score and integrated brier score are very bad (high)
# because of the extreme values in sale_time

performance_newcox_glmnet <- function(cvglmnet, 
                                      lambda_value="lambda.1se", 
                                      variables_names, 
                                      embedding_names="", 
                                      dataset,
                                      dataset_test,
                                      surv_call="Surv(sale_time, event"){
  #' This function gives the performance of a new cox model based on the important variable determined by a cv.glmnet object. Factor levels that are deleted by the algorithm are merged to the reference level. This function assumes that the train and test sets have the same factors for every categorical variables (which should be true if both dataset are big enough).
  #'
  #' @param cvglmnet cv.glmnet object
  #' @param lambda_value numeric. default="lambda.1se". Value of lambda to use in coef(cv.glmnet, s=lambda)
  #' @param variable_names str. Vector of strings with all the variables used in the Cox model, except the variables refering to the embeddings.
  #' @param embedding_names str (optional). Variables refering to the text embeddings.
  #' @param dataset data.frame or tibble. Dataset used to train the model used in cv.glmnet
  #' @param dataset_test data.frame or tibble. Dataset used to assess performance of cox model. This function simply applies the same transformations that are applied to `dataset`.
  #' @param surv_call str. Survival formula of the cox model in string format. e.g. "Surv(time1, time2, event)"
  #'
  #' @return scores
  
  newcox <- generate_newcox_glmnet(cvglmnet = cvglmnet
                                   ,variables_names=variables_names
                                   ,embedding_names=embedding_names
                                   ,dataset = dataset
                                   ,dataset_test = dataset_test
                                   ,surv_call="Surv(sale_time, event)"
  )
  
  scores <- performance_measures(model=newcox$cox
                                 ,olddata = newcox$train_df # or dataset
                                 ,newdata = newcox$test_df # or dataset_test
  )
  
  # extract number of variable related to embeddings selected by glmnet
  all_variables <- newcox$cox$coefficients %>% names() # variables selected by glmnet
  pattern1 = paste0(embedding_names, collapse = "$|") # exact match on embeddings names
  
  # each variable selected by glmnet is compared to the exact pattern
  # if there is no match, it returns character(0), if there is a match,
  # it returns the variable. I then unlist to remove character(0) to get a 
  # vector of variables related to text kept by glmnet, and I then compute
  # the length of the 
  number_text_variables <- str_extract_all(string=all_variables, pattern=pattern1) %>%
    unlist() %>%
    length()
  number_non_text_variables <- length(all_variables) - number_text_variables
  
  return(list(model=newcox
              ,scores=scores
              ,number_of_variables=list(non_text=number_non_text_variables, 
                                        text=number_text_variables)
              )
        )
}


