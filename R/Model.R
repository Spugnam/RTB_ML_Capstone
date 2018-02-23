
####################################
################Sito################
####################################


####################
#Load
####################
# feather load
library(feather)
impressions = as.data.table(read_feather("../data/split_by_day/concat/impressions.feather"))

# Check load
dim(impressions)

library('data.table')
class(impressions) # still a data.frame


####################
#Basic EDA
####################

# summary(impressions)
# View(cor(impressions[, numeric_columns, with=F]))

# Inspect numeric columns
# numeric_columns <- sapply(impressions, is.numeric)
# head(impressions[, numeric_columns, with=F])


####################
#Pre-Processing
####################

####################
# Train-Test Split
####################
View(head(impressions))

levels(impressions$day)

impressions[, .(Count = .N), by = TrainTestFlag] # conflict data.table/ tbl_df

impressions[, day := as.factor(day)]
setkey(impressions, day)
train = impressions[as.character(1:15)]
test = impressions[as.character(16:22)] # test ~ 20%  TODO: do not subsample test
dim(train)[1]+dim(test)[1] == dim(impressions)[1] # TRUE

train$day <- NULL
test$day <- NULL

###################
# Save data to file

# write train and test to csv
# write.csv(train, "./data/train.csv")
# write.csv(test, "./data/test.csv")
# mini_train = train[1:10]
# write.csv(mini_train, file="./mini_train.csv")
# mini_test = test[, clicked:= NULL]
# write.csv(mini_test[1:10], file="./data/mini_test.csv")


########################
#random intercept models
########################
library(lme4)
citation("lme4")

# build formula
iabs <- c('IAB1', 'IAB2', 'IAB3', 'IAB4',
          'IAB5', 'IAB6', 'IAB7', 'IAB8', 'IAB9', 'IAB10', 'IAB11',
          'IAB12', 'IAB13', 'IAB14', 'IAB15', 'IAB16', 'IAB17', 'IAB18',
          'IAB19', 'IAB20', 'IAB21', 'IAB22', 'IAB23', 'IAB24', 'IAB25') # dropping IAB26

f <- as.formula(paste0("clicked ~ ", 
                                 paste(iabs, collapse = " + "),
                                 " + (1|ad)" ))
f
basic.model <- glmer(f, train, REML=FALSE)


f.age <- as.formula(paste0("clicked ~ age +", 
                       paste(iabs, collapse = " + "),
                       " + (1|ad) " )) # + (1|timestamp_weekday)
f.age
basic.age.model <- glmer(f.age, 
                         data = train, 
                         family = binomial, 
                         control = glmerControl(optimizer = "bobyqa"),      
                         nAGQ = 1) 

# Model failed to converge: degenerate  Hessian with 2 negative eigenvalues
# https://stats.stackexchange.com/questions/242109/model-failed-to-converge-warning-in-lmer
# try marginally simpler model to intercept uncorrelated random intercepts/ slopes of age for each ad



summary(basic.age2.model) 

# Compare 2 models
anova(basic.model, basic.age.model) 
# Conclusion: age affected click 'value' (χ2(1)=54.855, p=1.298e-13), lowering it by 
# about -1.392e-05 ± 1.880e-06 (standard errors) (ok for logistic regression?)

# show coefficients for each ad
coef(basic.age.model)


########################
#random slope models
########################
f.age <- as.formula(paste0("clicked ~ age +", 
                           paste(iabs, collapse = " + "),
                           " + (age|ad) " )) # + (1|timestamp_weekday)
f.age

# simpler model
# f.age2 <- as.formula(paste0("clicked ~ age +", 
#                             paste(iabs, collapse = " + "),
#                             " + (0 + age |ad) + (1|ad) " )) # + (1|timestamp_weekday)

age.model <- glmer(f.age, 
                   data = train, 
                   family = binomial, 
                   control = glmerControl(optimizer = "bobyqa"),      
                   nAGQ = 0) 

summary(age.model)
coef(basic.age.model) # different intercept and slope for age per ad


# Predictions
test_cleaned <- test[,c('age', 'timestamp_hour', 'timestamp_weekday', 'ad', iabs), with=F]
predictions <- predict(age.model, newdata = test_cleaned, type = "response")

###################
# Model evaluation: 

# Feature importance

# ROC
library(ROCR)
# Compute AUC for predicting Class with the model
prediction_object <- prediction(predictions, test$clicked)
perf <- performance(prediction_object, measure = "tpr", x.measure = "fpr")
plot(perf)
abline(0, 1, lty = "dotted")
auc <- performance(prediction_object, measure = "auc")
auc <- auc@y.values[[1]]
auc # 0.6297916

# Log loss/Cross-Entropy Loss
library(MLmetrics)
LogLoss(predictions, test$clicked) # 0.4198258

library(ggplot2)
ggplot(test_cleaned, aes(test_cleaned$age + runif(length(test_cleaned), 0, .7),
                     predictions+ runif(length(test_cleaned), 0, .01))) + 
  geom_point(alpha = 1/40) +
  coord_cartesian(xlim = c(0, 60))
# Conclusion: hard to interpret graph as is

# Better way to interprete: average marginal probability (See https://stats.idre.ucla.edu/r/dae/mixed-effects-logistic-regression/)
# graph the average change in probability of the outcome across the range of some predictor of interest (age)
# for all groups (ads) in our sample

mean_quantiles <-  function(x) {
  c(M = mean(x), quantile(x, c(0.25, 0.75)))
}

predictions <- lapply(1:80, function(j) {
  test_cleaned$age <- j
  predict(age.model, newdata = test_cleaned, type = "response")
})

# plot average marginal predicted probabilities
mean_age_predictions <- t(sapply(predictions, mean_quantiles))
plotdata <- as.data.frame(cbind(mean_age_predictions, 1:80))
colnames(plotdata) <- c("PredictedProbability", "Lower", "Upper", "Age")

# plot average marginal predicted probabilities
ggplot(plotdata, aes(x = Age, y = PredictedProbability)) + geom_line() 
#+ylim(c(0, 1))

ggplot(plotdata, aes(x = Age, y = PredictedProbability)) + 
  geom_linerange(aes(ymin = Lower, ymax = Upper)) + 
  geom_line(size = 2) + ylim(c(0.05, .2))

#####################################################################################
test_cleaned_mini = test_cleaned[0:10000]

# calculate predicted probabilities and store in a list
biprobs <- lapply(levels(test_cleaned_mini$ad), function(ad_) {
  test_cleaned_mini$ad <- ad_
  lapply(1:80, function(j) {
    test_cleaned_mini$age <- j
    predict(age.model, newdata = test_cleaned_mini, type = "response")
  })
})

# get means and quartiles for all ages for each ad
plotdata2 <- lapply(biprobs, function(X) {
  temp <- t(sapply(X, function(x) {
    c(M=mean(x), quantile(x, c(.25, .75)))
  }))
  temp <- as.data.frame(cbind(temp, 1:80))
  colnames(temp) <- c("PredictedProbability", "Lower", "Upper", "Age")
  return(temp)
})

# collapse to one data frame
plotdata2 <- do.call(rbind, plotdata2)

# add ad
plotdata2$ad <- factor(rep(levels(test_cleaned$ad), each = length(1:80)))

# show first few rows
head(plotdata2)

# plot
ggplot(plotdata2, aes(x = Age, y = PredictedProbability)) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper, fill = ad), alpha = .15) +
  geom_line(aes(colour = ad), size = 2) + facet_wrap(~  ad)
# +  ylim(c(0.05, 0.15))



# lme4 performance tricks:
# https://cran.r-project.org/web/packages/lme4/vignettes/lmerperf.html




