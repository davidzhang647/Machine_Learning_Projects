######## load libraries
library(tidyverse)
library(corrplot)
library(fastDummies)
library(rpart)
library(rpart.plot)
library(caret)
library(glmnet)
library(randomForest) 
library(ROCR)
library(pROC)
library(naivebayes)
library(xgboost)
library(e1071)
library(vegan)
library(factoextra)
library(kableExtra)
library(jtools)
library(inspectdf)
library(cowplot)

######### load the data set
bank <- read.csv("project/bank.csv")


##########################################
## Exploratory Data Analysis
##########################################

## Overview of categorical variables
x <- inspect_cat(bank, show_plot = T)
show_plot(x, text_labels = T)

## Distributions of deposit in numbers and percentages
## dataset is balanced. No need to do resampling.
deposit_number <- bank %>% 
  group_by(deposit) %>% 
  summarize(Count = n()) %>%
  ggplot(aes(x = deposit, y = Count)) + 
  geom_bar(stat = "identity", fill = "#b779ed", color = "grey40") + 
  theme_bw() + 
  coord_flip() + 
  geom_text(aes(x = deposit, y = 0.01, label = Count),
            hjust = -0.8, vjust = -1, size = 3, 
            color = "black", fontface = "bold") + 
  labs(title = "Deposit", x = "Deposit", y="Amount") + 
  theme(plot.title=element_text(hjust=0.5))

deposit_percentage <- bank %>% group_by(deposit) %>% summarise(Count=n()) %>% 
  mutate(pct = round(prop.table(Count), 2) * 100) %>% 
  ggplot(aes(x=deposit, y=pct)) + 
  geom_bar(stat = "identity", fill = "#62dce3", color="grey40") + 
  geom_text(aes(x=deposit, y=0.01, label= sprintf("%.2f%%", pct)),
            hjust=0.5, vjust=-3, size=4, 
            color="black", fontface = "bold") + 
  theme_bw() + 
  labs(x = "Deposit", y="Percentage") + 
  labs(title = "Deposit (%)") + theme(plot.title=element_text(hjust=0.5))

plot_grid(deposit_number, deposit_percentage, align="h", ncol=2)

## Deposit Subscriptions based on Education Level
bank %>%
  group_by(education, deposit) %>%
  tally() %>% 
  ggplot(aes(x = education, y = n, fill = deposit)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Education Level", y = "Number of People") +
  ggtitle("Deposit Subscriptions Based on Education Level") +
  geom_text(aes(label = n), vjust = -0.5, position = position_dodge(0.8))

## Deposit Subscriptions based on Marital Status
bank %>%
  group_by(marital, deposit) %>%
  tally() %>% 
  ggplot(aes(x = marital, y = n, fill = deposit)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Marital Status", y = "Number of People") +
  ggtitle("Deposit Subscriptions based on Marital Status") +
  geom_text(aes(label = n), vjust = -0.5, position = position_dodge(0.8))

## Deposit Subscriptions Based on Last Contact Duration(in seconds)
bank %>%
  group_by(duration, deposit) %>%
  tally() %>% 
  ggplot(aes(x = duration, y = n, color = deposit)) +
  geom_smooth(se = F) +
  labs(x = "Duration(in seconds)", y = "Number of People") +
  ggtitle("Deposit Subscriptions Based on Last Contact Duration") 

## Deposit Subscriptions based on jobs
bank %>%
  group_by(job, deposit) %>%
  tally() %>% 
  ggplot(aes(x = job, y = n, fill = deposit)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Job", y = "Number of People") +
  ggtitle("Deposit Subscriptions Based on Jobs") +
  geom_text(aes(label = n), vjust = -0.5, position = position_dodge(0.8))

## Changes in Deposit Subscriptions vs Age vs Personal Loans
bank %>% 
  group_by(age, deposit, loan) %>% 
  tally() %>% 
  ggplot(aes(x = age, y = n, color = loan)) +
  geom_smooth(se=F) +
  labs(title = "Changes in Deposit Subscriptions vs Age vs Personal Loans",
       x = "Age",
       y = "Number of People")

## Changes in Deposit Subscriptions vs Age vs Contact Methods
bank %>% 
  group_by(contact, age, deposit) %>% 
  tally() %>% 
  ggplot(aes(x = age, y = n, color = contact)) +
  geom_smooth(se=F) +
  labs(title = "Changes in Deposit Subscriptions vs Age vs Contact Methods",
       x = "Age",
       y = "Number of People")


##########################################
## Data Cleaning
##########################################

## remove unnecessary columns - month and day
bank2 <- bank %>%
  filter(!(pdays != -1 & poutcome=='unknown')) %>%
  filter(!(job == "unknown")) %>% 
  select(-month, -day, -poutcome)


dim(bank2)  ## 11090 x 15
skimr::skim(bank2)
View(bank2)


bank_clean <- fastDummies::dummy_cols(bank2, remove_first_dummy = T) %>% 
  select(-contact, -default, -deposit,
         -education, -housing, -job,
         -loan, -marital, -job_unknown)

colnames(bank_clean) <- sub("-","_", colnames(bank_clean))

bank_clean$train <- NULL
saveRDS(bank_clean, "project/bank_clean.RDS")

## Split into training/test dataset
set.seed(820)

# Determine sample size
bank_clean$train <- sample(c(0,1), nrow(bank_clean), replace = TRUE, prob = c(0.3, 0.7))

# train dataset - 7731 observations
bank_train <- bank_clean %>% filter(train == 1)
bank_train$train <- NULL

# test dataset - 3359 observations
bank_test <- bank_clean %>% filter(train == 0)
bank_test$train <- NULL

########################################################
## Logistic regression
########################################################
logit <- glm(deposit_yes~., data=bank_train, family="binomial")

yhat_logit <- predict(logit, bank_test,type='response')

modelroc_logit <- roc(as.ordered(y_test), yhat_logit)
modelroc_logit

# AUC = 0.867
plot(modelroc_logit, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)

########################################################
## Lasso
########################################################

x_train <- model.matrix(deposit_yes ~ ., bank_train)[, -1]
y_train <- bank_train$deposit_yes
x_test <- model.matrix(deposit_yes ~ ., bank_test)[ ,-1] 
y_test <- bank_test$deposit_yes

fit_lasso <- cv.glmnet(x_train, y_train, alpha = 1,
                       nfolds = 10, family="binomial")

yhat_lasso_test <- predict(fit_lasso, x_test, s = fit_lasso$lambda.min)

modelroc_lasso <- roc(as.ordered(y_test), yhat_lasso_test)
modelroc_lasso

# AUC = 0.867
plot(modelroc_lasso, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)


# Coef
coef(fit_lasso)

########################################################
## Random Forest
########################################################
rf <- randomForest(deposit_yes ~ ., data = bank_train, ntree = 500)

yhat_rf_test <- predict(rf, bank_test[, -27], type='response')

varImpPlot(rf)

modelroc_rf <- roc(as.ordered(y_test), as.ordered(yhat_rf_test))
modelroc_rf

# AUC = 0.887
plot(modelroc_rf, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)

########################################################
## Boosting
########################################################

train_bst <- bank_train %>%  dplyr::select(-deposit_yes)
test_bst <- bank_test %>%  dplyr::select(-deposit_yes)

test_m <- as.matrix(test_bst)
train_m <- as.matrix(train_bst)
deposit_train <- bank_train$deposit_yes
deposit_test <- bank_test$deposit_yes

# here we choose 0.01 instead of 0.001
# to avoid overfitting controlling 1000 rounds. 
bst <- xgboost(data = train_m, label = deposit_train, eta =0.01,
               max_depth = 6, nrounds = 800, objective = "binary:logistic")
pred <- predict(bst,test_m, type='response')

modelroc_boost <- roc(as.ordered(y_test), pred)
modelroc_boost

# AUC = 0.891
plot(modelroc_boost, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)

summary(bst)
########################################################
## Support Vector Machines
########################################################
svm_model <- svm(deposit_yes ~ ., data = bank_train, kernel="linear", scale = T)

yhat_svm_test <- predict(svm_model, bank_test[, -27])

modelroc_svm <- roc(as.ordered(y_test), yhat_svm_test)
modelroc_svm

# AUC = 0.862
plot(modelroc_svm, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)
  

########################################################
## Naive Bayes
########################################################
bank_train_nb <- bank_train
bank_train_nb$deposit_yes <- as.factor(bank_train_nb$deposit_yes)

nb <- naive_bayes(deposit_yes ~ ., data = bank_train_nb)

yhat_nb_test <- predict(nb, bank_test[, -27], type = "prob")

modelroc_nb <- roc(as.ordered(y_test), yhat_nb_test[, 2])
modelroc_nb

# AUC = 0.781
plot(modelroc_nb, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)

########################################################
## Models Measurement
########################################################

auc_table <- tibble(Model = c("Logistic",
                      "Lasso",
                      "Random Forest",
                      "Boosting",
                      "SVM",
                      "Naive Bayes"),
                      AUC = c(modelroc_logit$auc,
                        modelroc_lasso$auc,
                        modelroc_rf$auc,
                        modelroc_boost$auc,
                        modelroc_svm$auc,
                        modelroc_nb$auc))

ggplot(auc_table, aes(x = Model, y = AUC)) +
  geom_bar(aes(fill = Model), stat = "identity") +
  geom_text(aes(label = round(AUC, 4)), vjust=1.6, color="white", size=3.5) +
  coord_cartesian(ylim = c(0.75, 0.9)) +
  scale_fill_manual(values = c("#0072B2", "#999999","dark grey", "#D55E00","#56B4E9", "purple")) +
  theme_minimal()


########################################################
## Clustering
########################################################
library(analogue) 
library(sets)
data <- bank_clean

data_n <- data[, 1:6]
data_c <- data[, 7:27]

data_nscale = scale(data_n)

distc <- vegdist(data_c, method = "jaccard")
distn <- vegdist(data_nscale,method = "euclidean")
complete_dis <- fuse(distc,distn, weights = c(0.5,0.5))

dm <- as.matrix(complete_dis)

hward <- hclust(complete_dis, method = "ward.D")
plot(hward)

table(cutree(hward,k = 3))
cutree(hward,k = 3) -> grouping

fviz_cluster(list(data = data, cluster = grouping))

fviz_nbclust(data, FUN = hcut, method = "wss")
fviz_nbclust(data, FUN = hcut, method = "silhouette")

data$cluster = cutree(hward, k = 3)
data$cluster1 = cutree(hward, k = 4)
data$cluster2 = cutree(hward, k = 2)

cluster_s <- data %>% 
  group_by(cluster) %>% 
  skimr::skim_to_wide() %>% 
  select(cluster, variable, mean) %>% 
  mutate_at(vars(mean), as.numeric)

cluster_dat = cluster_s%>% 
  pivot_wider(names_from = variable, 
              values_from=mean)

heatmap(as.matrix(cluster_dat), 
        scale="column", 
        Colv = NA, 
        Rowv = NA)

data %>%
  group_by(cluster) %>%
  skimr::skim_to_wide()%>%
  summary()

data %>%
  group_by(cluster) %>%
  summarize(cluster_age = mean(age),
            cluster_balance = mean(balance),
            cluster_campaign = mean(campaign),
            cluster_days = mean(pdays),
            cluster_duration = mean(duration),
            cluster_previous = mean(previous))

# ggplot(data, aes(age, balance, group = cluster, color = factor(cluster))) + 
#   geom_point(alpha = 0.5,position = "jitter") +
#   scale_color_manual(values = c("#c41414","#ffb92e","#6ca5f5",#1b9400","#006bc2")) +
#   theme_bw()

ggplot(data, aes(factor(cluster),age,fill = factor(cluster))) +
  geom_boxplot() +
  scale_fill_manual(values = c("#c41414","#ffb92e","#6ca5f5","#1b9400","#006bc2")) +
  theme_bw() +
  labs(title = "Age Distribution by Cluster")

ggplot(data, aes(balance, fill = factor(cluster))) +
  geom_histogram(position = "dodge") +
  facet_wrap(~factor(cluster)) +
  scale_fill_manual(values = c("#c41414","#ffb92e","#6ca5f5","#1b9400","#006bc2")) +
  theme_bw() +
  labs(title = "Balance Distribution by Cluster")


dend <- as.dendrogram(hward)
dend %>% 
  set("labels_col", value = c("#c41414","#ffb92e","#6ca5f5","#1b9400","#006bc2"), k = 3) %>% 
  plot(horiz = F, axes=F, main = "Visual Breakdown of Three Clusters")

