# Import libraries
library(dplyr)
library(tidyquant)
library(cluster)
library(caret)
library(rpart)
library(rpart.plot)
library(FNN)

# Get the SP500 stocks using the tidyquant library
sp500 <- tq_index("SP500")
sp500_data <- tq_get(sp500$symbol, from = "2021-09-01", to = "2024-09-01", get = "stock.prices")

sp500_data<-sp500_data[,c(1,2,7,8)]

# Obtain the logarithm of the adjusted price
sp500_data$log_price<-cbind(log(sp500_data$adjusted))

# Compute first differences of the log price to get the return
sp500_diff <- sp500_data %>%
  group_by(symbol) %>%
  arrange(date) %>%
  mutate(r = log_price - lag(log_price))

# Get a new df with the mean and sd of the stocks
sp500_summary <- sp500_diff %>%
  group_by(symbol) %>%
  summarize(
    mean_diff = mean(r, na.rm = TRUE),
    sd_diff = sd(r, na.rm = TRUE)
  )

#Normalize data
normalization<-preProcess(sp500_summary[,-1],method=c("center","scale"))
sp500_nrmsum <- as.data.frame(predict(normalization, sp500_summary))

# Set seed for reproducibility
set.seed(123)

#Create data frame to compare each k
results.df<-data.frame(k=seq(1:10), sil=0)


# Perform k-means clustering from k=1 to 10 and save variance and silouette
for (k in (1:10)){
  kmeans_result <- kmeans(sp500_nrmsum[, -1],k,nstart=30)
  if (k==1){
    results.df[k,2] <- 0
  }
  else{
    results.df[k,2] <- mean(silhouette(kmeans_result$cluster, dist(sp500_nrmsum[,-1]))[,3])
  }
}

# Plot silhouette score vs k
ggplot(data=results.df, aes(k,sil))+
  geom_line()+
  theme_classic()+
  geom_text(aes(label = k), vjust = -1, hjust = 0.5) +
  labs(title = "Silhouette score vs k", x = "k", y = "Silhouette score")

# Plot elbow method
fviz_nbclust(sp500_nrmsum[,-1], kmeans, method="wss")

#Plot clusters with k=3
cluster<-kmeans(sp500_nrmsum[, -1],3,nstart=30)
fviz_cluster(cluster, data=sp500_nrmsum[,-1], palette=c("blue","red","green"))+
  theme_classic()+
  labs(title = "Stocks Clustering", x = "Return", y = "Volatility")


#Add cluster column in common and normalized datasets
sp500_nrmsum$cluster<-cbind(cluster$cluster)
sp500_summary$cluster<-cbind(cluster$cluster)


sp500_nrmsum$cluster<-as.factor(sp500_nrmsum$cluster)
sp500_summary$cluster<-as.factor(sp500_summary$cluster)

# Plot centroids
sp500_clust1<-filter(sp500_summary,cluster==1)
sp500_clust2<-filter(sp500_summary,cluster==2)
sp500_clust3<-filter(sp500_summary,cluster==3)

cluster_lt<-data.frame(cluster=c(1,2,3),return=c(mean(sp500_clust1$mean_diff, na.rm = TRUE),mean(sp500_clust2$mean_diff, na.rm = TRUE),mean(sp500_clust3$mean_diff, na.rm = TRUE)),volatility=c(mean(sp500_clust1$sd_diff, na.rm = TRUE),mean(sp500_clust2$sd_diff, na.rm = TRUE),mean(sp500_clust3$sd_diff, na.rm = TRUE)))

color_cluster <- c(`1`  = "blue",
                   `2` = "red",
                   `3` = "green" )

ggplot(cluster_lt,aes(x=return,y=volatility,color=as.factor(cluster)))+
  geom_point()+
  labs(title="Clusters Volatility vs Return",x="Return",y="Volatility",color="Cluster")+
  scale_color_manual(values=color_cluster)+
  theme_minimal()


#Plot clusters density functions
ggplot(sp500_summary, aes(x = mean_diff, fill = cluster)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("1" = "lightblue", "2" = "lightpink", "3" = "lightgreen")) +
  labs(title = "Clusters Return Density", x = "Return", y = "Probability Density") +
  theme_minimal()

#Create random sample for train and validation sets

sp500_summary$Volatility<-sp500_summary$sd_diff
sp500_summary$Return<-sp500_summary$mean_diff
set.seed(123)
train.index <- sample(c(1:dim(sp500_summary)[1]), dim(sp500_summary)[1]*0.6)
train.df <- sp500_summary[train.index, ]
valid.df <- sp500_summary[-train.index, ]

#Train the model

tree_mod <- rpart(cluster ~ Return+Volatility,
                  data = train.df,
                  method = "class",
                  maxdepth = 8,
                  cp = 0.001)
summary(tree_mod)

color_mapping <- c(`1`  = "lightblue",
                   `2` = "lightpink",
                   `3` = "lightgreen" )

#Plot the tree
prp(tree_mod,
    type = 1,
    extra = 1,
    under = TRUE,
    split.font = 1,
    varlen = -10,
    yesno = 2,
    box.col =color_mapping[as.character(tree_mod$frame$yval2[,1])])

#Prune the tree
prune.tree_mod <- prune(tree_mod, cp = tree_mod$cptable[which.min(tree_mod$cptable[,"xerror"]),"CP"])



#Plot the pruned tree
prp(prune.tree_mod,
    type = 1,
    extra = 1,
    under = TRUE,
    split.font = 1,
    varlen = -10,
    yesno = 2,
    box.col = color_mapping[as.character(prune.tree_mod$frame$yval2[,1])])

tree_result<-data.frame(actual = valid.df$cluster, predicted = predict(prune.tree_mod,valid.df[,-4],type="class"))
cm<-confusionMatrix(as.factor(tree_result$predicted),as.factor(tree_result$actual))


train_nrm.df <- sp500_nrmsum[train.index, ]
valid_nrm.df <- sp500_nrmsum[-train.index, ]

accuracy.df<-data.frame(k=seq(1,20),accuracy=0)

for (i in 1:20){
  knn_i<-knn(
    train_nrm.df[,c(2,3)],
    valid_nrm.df[,c(2,3)],
    cl=train_nrm.df[,4],
    k=i
  )
  accuracy.df[i,2]<-confusionMatrix(knn_i,valid_nrm.df[,4])$overall[1]
}


new_index <- c(
  "MELI", "PDD", "MRVL", "ASML", "DASH", "TTD", "WDAY",
  "AZN", "DDOG", "CCEP", "TEAM", "ZS", "ILMN", "GFS",
  "MDB", "ARM"
)
new_data <- tq_get(new_index, from = "2021-09-01", to = "2024-09-01", get = "stock.prices")

new_data<-new_data[,c(1,2,7,8)]

# Obtain the logarithm of the adjusted price
new_data$log_price<-cbind(log(new_data$adjusted))

# Compute first differences of the log price to get the return
new_diff <- new_data %>%
  group_by(symbol) %>%
  arrange(date) %>%
  mutate(r = log_price - lag(log_price))

# Get a new df with the mean and sd of the stocks
new_summary <- new_diff %>%
  group_by(symbol) %>%
  summarize(
    mean_diff = mean(r, na.rm = TRUE),
    sd_diff = sd(r, na.rm = TRUE)
  )

new_summary$Volatility<-new_summary$sd_diff
new_summary$Return<-new_summary$mean_diff

new_prediction<-data.frame(stock=new_summary$symbol,class=predict(prune.tree_mod,new_summary, type="class"))
