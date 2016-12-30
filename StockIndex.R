rm(list=ls())

# create df of stock prices over time
# initial value chosen randomly, each percentage change also random
# re-weight at the end of each period by market cap

N_STOCKS <- 10
N_PERIODS <- 500
DIVISOR <- 10^6
seed <- floor(runif(1,0,10^7))
# print(paste("seed:",seed))
set.seed(seed)

initial_prices <- runif(N_STOCKS,5,50) # choose reasonable-enough prices
initial_shares <- (runif(N_STOCKS,100,1000))^2 # numbers of shares, leave static for now

get_price <- function(t,i,df)
  df[t,paste("p_",as.character(i),sep="")]

get_shares <- function(t,i,df)
  df[t,paste("n_",as.character(i),sep="")]

get_cap <- function(t,i,df) {
  col <- paste("c_",i,sep="")
  if (!is.na(df[t,col])) {
    return(df[t,col])
  }
  get_price(t,i,df)*get_shares(t,i,df)
}

get_total_cap <- function(t,df) {
  if (!is.na(df[t,"total_cap"])) {
    return(df[t,"total_cap"])
  }
  sum(sapply(1:N_STOCKS,function(i) get_cap(t,i,df)))
}

get_weight <- function(t,i,df) {
  col <- paste("w_",i,sep="")
  if (!is.na(df[t,col])) {
    return(df[t,col])
  }
  get_cap(t,i,df)/get_total_cap(t,df)
}

get_index_value <- function(t,df) {
  if (!is.na(df[t,"index_value"])) {
    return(df[t,"index_value"])
  }
  
  # weighted average
  # sum(sapply(1:N_STOCKS,function(i) get_price(t,i,df)*get_weight(t,i,df)))
  
  # S&P 500 approach
  sum(sapply(1:N_STOCKS,function(i) get_cap(t,i,df)))/DIVISOR
}

get_weighted_average_price <- function(t,df) {
  if (!is.na(df[t,"weighted_average_price"])) {
    return(df[t,"weighted_average_price"])
  }
  sum(sapply(1:N_STOCKS,function(i) get_price(t,i,df)*get_weight(t,i,df)))
}

#df <- data.frame(
#  p_1 = initial_prices[1],
#  n_1 = initial_shares[1],
#  #w_1 = get_weight(),
#  p_2 = initial_prices[2],
#  n_2 = initial_shares[2],
#  #w_2 = get_weight(),
#  p_3 = initial_prices[3],
#  n_3 = initial_shares[3],
#  #w_3 = get_weight(),
#  p_4 = initial_prices[4],
#  n_4 = initial_shares[4],
#  #w_4 = get_weight(),
#  p_5 = initial_prices[5],
#  n_5 = initial_shares[5],
#  #w_5 = get_weight(),
#  #total_cap = get_total_cap(1,df),
#  stringsAsFactors=FALSE
#)

initialize_df <- function(df) {
  df <- data.frame(matrix(ncol=4*N_STOCKS+3,nrow=N_PERIODS)) # no values initialized yet
  for (i in 1:N_STOCKS) {
    colnames(df)[4*i-3] <- paste("p_",i,sep="")
    colnames(df)[4*i-2] <- paste("n_",i,sep="")
    colnames(df)[4*i-1] <- paste("c_",i,sep="")
    df[1,paste("p_",i,sep="")] <- initial_prices[i]
    df[1,paste("n_",i,sep="")] <- initial_shares[i]
    df[1,paste("c_",i,sep="")] <- get_cap(1,i,df)
  }
  colnames(df)[4*N_STOCKS+1] <- "total_cap"
  df[1,"total_cap"] <- get_total_cap(1,df)
  for (i in 1:N_STOCKS) {
    colnames(df)[4*i] <- paste("w_",i,sep="")
    df[1,paste("w_",i,sep="")] <- get_weight(1,i,df)
  }
  colnames(df)[4*N_STOCKS+2] <- "index_value"
  colnames(df)[4*N_STOCKS+3] <- "weighted_average_price"
  df[1,"index_value"] <- get_index_value(1,df)
  df[1,"weighted_average_price"] <- get_weighted_average_price(1,df)
  return(df)
}

df <- initialize_df()

get_change_factor <- function()
  exp(rnorm(1,0,0.02))
  #exp(rcauchy(1,0,0.0001)) # kick it up a notch

update_price <- function(t_old,i,df)
  get_price(t_old,i,df)*get_change_factor()

update_shares <- function(t_old,i,df)
  get_shares(t_old,i,df)*1

update_df <- function(df) {
  t_old <- length(which(!is.na(df[,1]))) # number of periods filled in so far (in col 1)
  t <- t_old+1
  for (i in 1:N_STOCKS) {
    df[t,paste("p_",i,sep="")] <- update_price(t_old,i,df)
    df[t,paste("n_",i,sep="")] <- update_shares(t_old,i,df)
    df[t,paste("c_",i,sep="")] <- get_cap(t,i,df)
  }
  df[t_old+1,"total_cap"] <- get_total_cap(t,df)
  for (i in 1:N_STOCKS) {
    df[t,paste("w_",i,sep="")] <- get_weight(t,i,df)
  }
  df[t,"index_value"] <- get_index_value(t,df)
  df[t,"weighted_average_price"] <- get_weighted_average_price(t,df)
  return(df)
}

#get_total_cap(1,df)
#get_index_value(1,df)
#sapply(1:N_STOCKS,function(i) get_price(1,i,df))

for (i in 2:N_PERIODS) {
  df <- update_df(df)
}

# S&P method index graph (directly proportional to total market cap)
plot(1:N_PERIODS,df[,"index_value"],"l")

# weighted average price graph
plot(1:N_PERIODS,df[,"weighted_average_price"],"l")

# correlation of the index and the weighted avg price
cor(df[,"index_value"],df[,"weighted_average_price"])

# correlation of the changes in the index and the changes in the weighted avg price
cor(sapply(2:N_PERIODS,function(t) df[t,"index_value"]-df[t-1,"index_value"]),
    sapply(2:N_PERIODS,function(t) df[t,"weighted_average_price"]-df[t-1,"weighted_average_price"]))
