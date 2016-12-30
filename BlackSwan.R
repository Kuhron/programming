draw_addend <- function() {
  addend <- if (runif(1) < 0.5) 0 else 1
  while (runif(1) < 0.5) addend <- addend*2
  addend
}

get_random_integer <- function(precision, signed = TRUE) {
  len <- 1
  while(runif(1) < 1/precision) len <- len*precision
  c <- sample(0:len-1,1)
  if (signed && c != 0 && runif(1) < 0.5) c <- -c
  c
}

get_random_integer(5)

nums <- sapply(1:10^6, get_random_integer, precision=2)
plot(nums,type="l")

sums <- rep(NA,10^6)
sums[1] <- nums[1]
sums[2] <- nums[1]+nums[2]
sums <- sapply(2:10^6, function(n) sums[n-1]+nums[n])
plot(sums,type="l")

