# estimate someone's Kinsey scale value based on the number of guys and girls they've been attracted to
# if nonbinary is to be included here, try adding x to guys and y to girls such that x+y=1 and x/y reflects the gender balance

get_ci_z_score <- function(confidence_level) {
  # the z value is the distance from 0 in which CI% of the normal distribution is found
  # e.g. for CI = 95%, since -1.96 <= x <= 1.96 contains 95% of the normal distribution, Z = 1.96
  probability_mass_within_interval = confidence_level
  probability_mass_on_one_tail = (1-confidence_level)/2  # e.g. for 95% CI, need 2.5% on upper tail and 2.5% on lower tail
  quantile_at_upper_limit = 1 - probability_mass_on_one_tail
  quantile_at_lower_limit = probability_mass_on_one_tail
  
  # qnorm(quantile) gives you the value of x where {quantile} of the probability mass exists <= x
  # e.g. qnorm(0) = -Inf, qnorm(0.5) = 0, qnorm(1) = Inf
  z = qnorm(quantile_at_upper_limit)
  minus_z = qnorm(quantile_at_lower_limit)
  stopifnot(z == -1 * minus_z)  # assert
  return(z)
}

get_ci_binomial_probability_normal_approximation <- function(r, n, ci_confidence) {
  # NOTE: this uses the normal approximation, which is unreliable for small sample size or hen p is close to 0 or 1
  # about estimator for binomial probability: https://moonvalley.guhsdaz.org/common/pages/DisplayFile.aspx?itemId=17682112
  # also: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
  p_hat = r/n
  n_p_hat = n * p_hat
  n_q_hat = n * (1 - p_hat)
  if (n_p_hat <= 5 || n_q_hat <= 5) {
    stop(paste("not enough trials for good binomial probability estimation with r=",r,", n=",n, sep=""))
  }
  if (p_hat <= 0.05 || p_hat >= 0.95) {
    stop(paste("p too close to 0 or 1 for good binomial probability estimation with p_hat=",p_hat,sep=""))
  }
  z = get_ci_z_score(ci_confidence)
  ci_distance = z * sqrt(p_hat * (1-p_hat) / n)
  e = ci_distance
  return(c(p_hat - e, p_hat, p_hat + e))
}

get_ci_binomial_probability_wilson <- function(r, n, ci_confidence) {
  # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
  # has better properties, e.g. the ci's width doesn't go to zero when p goes to 0 or 1
  p_hat = r/n
  z = get_ci_z_score(ci_confidence)
  first_term = (1/(1+((z**2)/n))) * (p_hat + ((z**2) / (2*n)))
  radicand = ((p_hat * (1-p_hat))/n) + ((z**2)/(4*n**2))
  second_term = (z/(1+((z**2)/n))) * sqrt(radicand)
  return(c(first_term - second_term, p_hat, first_term + second_term))
}

get_kinsey_estimators <- function(n_hetero, n_homo, ci_confidence) {
  n = n_hetero + n_homo
  r = n_homo  # number of successes
  # binomial_probability_estimators = get_ci_binomial_probability_normal_approximation(r, n, ci_confidence)  # bad estimators
  binomial_probability_estimators = get_ci_binomial_probability_wilson(r, n, ci_confidence)
  return(6 * binomial_probability_estimators)
}

# Kinsey = 0 for all hetero, 6 for all homo
# so Kinsey = 6 * prob(homo)
n_hetero = 80
n_homo = 15
ci_confidence = 0.75

print(get_kinsey_estimators(n_hetero, n_homo, ci_confidence))