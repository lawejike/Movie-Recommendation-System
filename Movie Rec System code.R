library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

#creates a temporary file name
dl <- tempfile()

#download file from web to temp file
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

#read rating data file into a data table with assigned column names
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# extract movies (string) file into a matrix with 3 column and assign column names
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# for R 4.0 or later. convert movie matrix to a dataframe
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

# create Movielens data set from rating and movie dataframes
movielens <- left_join(ratings, movies, by = "movieId")

# Split Movielens into edx and validation set. Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# removes files and table that are no longer needed
rm(dl, ratings, movies, test_index, temp, movielens, removed)

##### CAPSTONE Project #####

# 1st step: generate train and test sets from edx

#set seed (i.e the start number used in random sequence generation)
set.seed(1, sample.kind = "Rounding")

#split edx into two sets, train and test
test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set_pre <- edx[test_index,]

# Make sure userId and movieId in test set are also in training set
test_set <- test_set_pre %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(test_set_pre, test_set)
train_set <- rbind(train_set, removed)

# remove datasets not needed
rm(test_set_pre, removed, test_index) 

# define function for computing RMSE for vectors of rating and corresponding predictors
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))}

# set number of digits to be printed 
options(digits = 5) 

#### Data Exploration ####
dim(edx) # number of rows and columns in edx
edx %>% head()  # print first 6 entries of edx
n_distinct(edx$movieId) # number of unique movies rated
n_distinct(edx$userId) # number of unique users that provide ratings

# plot histogram of the count of movie ratings
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 20, color = "black") + 
  scale_x_log10() + 
  ggtitle("Figure 1: Distribution of count of rated movies ") +
  xlab("count of movie rating") +
  ylab("frequency")

# plot histogram (frequency distribution) of the count of user ratings
edx %>%
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 20, color = "black") + 
  scale_x_log10() + 
  ggtitle("Figure 2: Distribution of counts of user ratings ") +
  xlab("count of ratings by users") +
  ylab("frequency")

#### Model 0: Using the average ####

# compute the average of all the ratings,mu_hat
mu_hat <- mean(train_set$rating)

# for simplicicty, use mu for mu_hat for the rest of code
mu <- mu_hat

# RMSE obtained by predicting mu for all unknown rating 
Model_0_rmse <- RMSE(test_set$rating, mu)

#print value of  naive_rmse
Model_0_rmse 

# set up table for storing various approaches and corresponding RMSE
RMSE_results <- data.frame(Approach = "Using the average", RMSE = Model_0_rmse)

#format RMSE column to 5 decimal place
RMSE_results$RMSE <- format(RMSE_results$RMSE,digits = 5) 
#print table
RMSE_results 


#### Model_1: accounting for movie effect(b_i) ####

# create a table of movieId and average bias for each movie
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu)) 

# predict rating as average rating plus movie bias
predicted_ratings <- mu + test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)   

#calculate and assign RMSE to Model 1
Model_1_rmse <- RMSE(predicted_ratings, test_set$rating) 

# Update RMSE table
RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Approach ="Movie Effect Model",
                                     RMSE = Model_1_rmse )) 
#print RMSE table
RMSE_results 



#### Model_2: account for user effect(b_u) ####

#compute and plot count vs average rating for user that rated over 100 movies
train_set %>%
  group_by(userId) %>%
  summarize(user_mean_rating = mean(rating)) %>%
  filter(n()>=100) %>%
  ggplot(aes(user_mean_rating)) +
  geom_histogram(bins = 35, color = "black") + 
  ggtitle("Figure 3: Distribution of average user ratings ") 

# create a table of userId and average bias for each user
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# predict rating as sum of average rating, movie bias, and user bias
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

#calculate and assign RMSE to Model 2
Model_2_rmse <- RMSE(predicted_ratings, test_set$rating)

# Update RMSE table
RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Approach ="Movie + User Effects Model",  
                                     RMSE = Model_2_rmse ))
#print RMSE table
RMSE_results



#### Model_3a: account for time effect (b_t) in WEEKS ####
# plot average rating vs date (in week) of rating
train_set %>% 
  mutate(dates = as_datetime(timestamp), date = round_date(dates, unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle( "Figure 4: Effect of time (in weeks) on rating")

# create a table of date (in week) and average time (date) bias 
week_avgs <- train_set %>% 
  mutate(dates = as_datetime(timestamp), date = round_date(dates, unit = "week")) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(date) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u))

# predict rating as sum of average rating, movie bias, user bias, and time (week) bias
predicted_ratings <- test_set %>%
  mutate(dates = as_datetime(timestamp), date = round_date(dates, unit = "week")) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(week_avgs, by='date') %>%
  mutate(pred = mu + b_i + b_u + b_t) %>%
  .$pred

#calculate and assign RMSE to Model 3a
Model_3a_rmse <- RMSE(predicted_ratings, test_set$rating)

# Update RMSE table
RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Approach ="Movie + User + Time (week) Effects Model",  
                                     RMSE = Model_3a_rmse ))  

#print RMSE table
RMSE_results

#### Model_3b: account for time effect (b_t) in MONTHS ####
# create a table of date (in month) and average time (date) bias
month_avgs <- train_set %>% 
  mutate(dates = as_datetime(timestamp), date = round_date(dates, unit = "month")) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(date) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u))

# predict rating as sum of average rating, movie bias, user bias, and time (month) bias 
predicted_ratings <- test_set %>%
  mutate(dates = as_datetime(timestamp), date = round_date(dates, unit = "month")) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(month_avgs, by='date') %>%
  mutate(pred = mu + b_i + b_u + b_t) %>%
  .$pred

#calculate and assign RMSE to Model 3b
Model_3b_rmse <- RMSE(predicted_ratings, test_set$rating)

# Update RMSE table
RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Approach ="Movie + User + Time (month) Effects Model",  
                                     RMSE = Model_3b_rmse ))  
#print RMSE table
RMSE_results

#### Model_3c: account for time effect (b_t) in YEAR ####
# create a table of date (in year) and average time (date) bias
year_avgs <- train_set %>% 
  mutate(dates = as_datetime(timestamp), date = round_date(dates, unit = "year")) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(date) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u))

# predict rating as sum of average rating, movie bias, user bias, and time (year) bias 
predicted_ratings <- test_set %>%
  mutate(dates = as_datetime(timestamp), date = round_date(dates, unit = "year")) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='date') %>%
  mutate(pred = mu + b_i + b_u + b_t) %>%
  .$pred

#calculate and assign RMSE to Model 3c
Model_3c_rmse <- RMSE(predicted_ratings, test_set$rating)

# Update RMSE table
RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Approach ="Movie + User + Time (year) Effects Model",  
                                     RMSE = Model_3c_rmse ))  

#print RMSE table
RMSE_results



#### Model_4: account for multi-genre effect (b_mg) ####
#generate table of genre,count and average rating of each genre
genre_rating <- train_set %>% 
  separate_rows(genres, sep = "\\|") %>% 
  group_by(genres) %>% 
  summarize(count = n(),rating = mean(rating))

# Print table with title
knitr::kable(genre_rating, caption = "Table 2: Average rating and count of unique genres in the train set")

# plot of genre vs average genre rating for train set
genre_rating %>% 
  ggplot(aes(genres, rating)) + 
  geom_point() + 
  geom_smooth() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Figure 5: Plot of genre vs average genre rating for train_set")

# count and average rating for multi-genres
train_set %>% group_by(genres) %>% 
  summarise(count = n(),rating = mean(rating)) %>% 
  top_n(20, count) %>% 
  arrange(desc(count)) %>% 
  knitr::kable(caption = " Table 3: Average rating and count of multi-genre in the train set")

# create a table of multi-genre and average multi-genre bias, b_mg
genre_avgs <- train_set %>%
  mutate(dates = as_datetime(timestamp), date = round_date(dates, unit = "week")) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(week_avgs, by='date') %>%
  group_by(genres) %>%
  summarize(b_mg = mean(rating - mu - b_i - b_u - b_t))

#predict rating as sum of average rating, movie bias, user bias, time (week) bias, and multi-genre bias
predicted_ratings <- test_set %>%
  mutate(dates = as_datetime(timestamp), date = round_date(dates, unit = "week")) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(week_avgs, by='date') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_t + b_mg) %>%
  .$pred

#calculate and assign RMSE to Model 4
Model_4_rmse <- RMSE(predicted_ratings, test_set$rating)

# Update RMSE table
RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Approach ="Movie + User + Time +  Multi-genre Effects Model",  
                                     RMSE = Model_4_rmse ))
# print RMSE table
RMSE_results



##### Regularization #####
# Movie bias and number of rating for the top 10 best movies 
train_set %>% count(movieId) %>% 
  left_join(movie_avgs) %>%
  arrange(desc(b_i)) %>% 
  select(movieId, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable(caption = "Table 4: Movie bias and number of rating for the top 10 best movies" )

# Movie bias and number of rating for the top 10 worst movies
train_set %>% count(movieId) %>% 
  left_join(movie_avgs) %>%
  arrange(b_i) %>% 
  select(movieId, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable(caption = "Table 5: Movie bias and number of rating for the top 10 worst movies")

# Note: Table 4 and 5 show the need for regularization

#### Model 5: Regularized movie effect model ####
#set range of value for lambda
lambdas <- seq(0, 10, 0.25)

#optimize lambda for movie
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  movie_reg_avgs <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  predicted_ratings <- test_set %>%
    left_join(movie_reg_avgs, by = "movieId") %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})

#plot lambdas vs rmses
qplot(lambdas, rmses)

# assign lambda_i to the lambda that gives minimum rmse
lambda_i <- lambdas[which.min(rmses)]

# print lambda
lambda_i

#print minimum rmse
min(rmses)

#A Use l to denote lambda. Assign l to 2.5 (the value obtained for lambda_i)
l <- 2.5
#B set average rating as mu
mu <- mean(train_set$rating)
#C create a table of movieId and regularized average bias(b_i) for each movie
movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))
#D predict rating as average rating(mu) plus regularized movie bias(b_i)
predicted_ratings <- test_set %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i ) %>%
  .$pred
#E calculate and assign RMSE to Model 5
Model_5_rmse <- RMSE(predicted_ratings, test_set$rating)

# Note: The set of codes in the previous 5 comments (#A,#B,#C,#D & #E) give the same
# value as the min RMSE obtained from optimization. These steps will be bypassed hereafter
# by simply assigning min RMSE as the RMSE of the model being considered

# Update RMSE table
RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Approach ="Regularized Movie Effects Model",  
                                     RMSE = Model_5_rmse ))
#print RMSE table
RMSE_results



#### Model 6: Regularized movie + user effect model ####
#set range of value for lambda
lambdas <- seq(0, 10, 0.25)

#optimize lambda for users with lambda_i = 2.5
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  movie_reg_avgs <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + 2.5))
  user_reg_avgs <- train_set %>% 
    left_join(movie_reg_avgs, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- test_set %>% 
    left_join(movie_reg_avgs, by = "movieId") %>%
    left_join(user_reg_avgs, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

#plot lambdas vs rmses
qplot(lambdas, rmses)

# assign lambda to the value that gives minimum rmse
lambda_u <- lambdas[which.min(rmses)]

#print lambda
lambda_u

#assign minimum rmse to Model 6
Model_6_rmse <-min(rmses)


# Update RMSE table
RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Approach ="Regularized Movie + User Effects Model",  
                                     RMSE = Model_6_rmse ))
# print RMSE table
RMSE_results



#### Model 7: Regularized movie + user + time effect model ####
#set range of value for lambda
lambdas <- seq(0, 10, 0.25)

# Add date column to the datasets outside of 'sapply' and reduce iteration time during optimization
train_set_d <- train_set %>%
  mutate(dates = as_datetime(timestamp), date = round_date(dates, unit = "week"))  
test_set_d <- test_set %>%
  mutate(dates = as_datetime(timestamp), date = round_date(dates, unit = "week")) 

#Assign lambda_i = 2.5, lambda_u = 5, and optimize lambda for time (date) 
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set_d$rating)
  movie_reg_avgs <- train_set_d %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+2.5))
  user_reg_avgs <- train_set_d %>%
    left_join(movie_reg_avgs, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+5))
  time_reg_avgs <- train_set_d %>%
    left_join(movie_reg_avgs, by='movieId') %>%
    left_join(user_reg_avgs, by='userId') %>%
    group_by(date) %>%
    summarize(b_t = sum(rating - mu - b_i - b_u)/(n()+l))
  
  predicted_ratings <- test_set_d %>%
    left_join(movie_reg_avgs, by='movieId') %>%
    left_join(user_reg_avgs, by='userId') %>%
    left_join(time_reg_avgs, by='date') %>%
    mutate(pred = mu + b_i + b_u + b_t) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

# plot lambdas vs rmses
qplot(lambdas, rmses)

# assign lambda_t as lambda that gives the minimum rmse
lambda_t <- lambdas[which.min(rmses)]

#print lambda
lambda_t

#assign minimum rmse to Model 7
Model_7_rmse <-min(rmses)

# Update RMSE table
RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Approach ="Regularized Movie + User + Time Effects Model",  
                                     RMSE = Model_7_rmse ))

#print RMSE table
RMSE_results

#### Model 8: Regularized movie + user + time + multi-genre model ####
#set range of value for lambda
lambdas <- seq(0, 10, 0.25)

# assign lambda_i = 2.5, lambda_u = 5,lambda_t = 2.5 and optimize lambda for multi-genre 
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set_d$rating)
  movie_reg_avgs <- train_set_d %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+2.5))
  user_reg_avgs <- train_set_d %>%
    left_join(movie_reg_avgs, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+5))
  time_reg_avgs <- train_set_d %>%
    left_join(movie_reg_avgs, by='movieId') %>%
    left_join(user_reg_avgs, by='userId') %>%
    group_by(date) %>%
    summarize(b_t = sum(rating - mu - b_i - b_u)/(n()+2.5))
  m_genre_reg_avgs <- train_set_d %>%
    left_join(movie_reg_avgs, by='movieId') %>%
    left_join(user_reg_avgs, by='userId') %>%
    left_join(time_reg_avgs, by='date') %>%
    group_by(genres) %>%
    summarize(b_mg = sum(rating - mu - b_i - b_u - b_t)/(n()+l))
  
  predicted_ratings <- test_set_d %>%
    left_join(movie_reg_avgs, by='movieId') %>%
    left_join(user_reg_avgs, by='userId') %>%
    left_join(time_reg_avgs, by='date') %>%
    left_join(m_genre_reg_avgs, by='genres') %>%
    mutate(pred = mu + b_i + b_u + b_t + b_mg) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

#plot lambdas vs rmses
qplot(lambdas, rmses)

#assign lambda_mg as lambda that gives minimum rmse
lambda_mg <- lambdas[which.min(rmses)]

# print lambda
lambda_mg

#assign minimum rmse to Model 8
Model_8_rmse <-min(rmses)

# Update RMSE table
RMSE_results <- bind_rows(RMSE_results,
                          data.frame(Approach ="Regularized Movie + User + Time + Multi-genre Effects Model",  
                                     RMSE = Model_8_rmse ))


#### Prediction on validation set using Model 8 and edx dataset ####

#set average rating on edx as mu
mu <- mean(edx$rating)

# create a table of movieId and regularized average movie bias
movie_avg <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n() + 2.5))

# create a table of userId and regularized average user bias
user_avg <- edx %>%
  left_join(movie_avg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n() + 5))

# create a table of dates and regularized average time bias
time_avg <- edx %>%
  mutate(dates = as_datetime(timestamp), date = round_date(dates, unit = "week")) %>%
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  group_by(date) %>%
  summarize(b_t = sum(rating - mu - b_i - b_u)/(n() + 2.5))

# create a table of genre and regularized average multi-genre bias
m_genre_avg <- edx %>%
  mutate(dates = as_datetime(timestamp), date = round_date(dates, unit = "week")) %>%
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  left_join(time_avg, by='date') %>%
  group_by(genres) %>%
  summarize(b_mg = sum(rating - mu - b_i - b_u - b_t)/(n() + 2))

# predict rating as sum of average rating, and regularized values of movie bias, 
# user bias, time(week) bias, and multi-genre bias
predicted_ratings <- validation %>%
  mutate(dates = as_datetime(timestamp), date = round_date(dates, unit = "week")) %>%
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  left_join(time_avg, by='date') %>%
  left_join(m_genre_avg, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_t + b_mg) %>%
  .$pred

# calculate RMSE based of predicted rating. Assign to rmse_val
rmse_val <-RMSE(predicted_ratings, validation$rating)

#print rmse
rmse_val