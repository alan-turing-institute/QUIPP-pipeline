library(mvtnorm)
library(mice)
library(ggplot2)

Nsample = 20
Nsynth = 100

## Ncol = 3
## L = matrix(c(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), ncol=Ncol)
## C = L %*% t(L)

## Observed samples
Ncol = 2
C = matrix(c(1.0, 0.9, 0.9, 1.0), ncol=Ncol)

samples = rmvnorm(Nsample, rep(0.0, Ncol), C)
samples = data.frame(samples)

## Specify missing data for the unsampled (synthetic) population
samples[(1 + Nsample):(Nsample + Nsynth),] = NA


## Predictive mean matching
## samples_imp = mice(samples, method="pmm")

## Bayesian linear regression
## samples_imp = mice(samples, method="norm")

## Bayesian bootstrap
samples_imp = mice(samples, method="norm.boot")

samples_completed = complete(samples_imp)

## Is this row synthetic? Useful for plotting
samples_completed[1:Nsample, "synth"] = FALSE
samples_completed[(1 + Nsample):(Nsample + Nsynth), "synth"] = TRUE

## Plot vs original data
g <- ggplot(samples_completed, aes(x=X1, y=X2, shape=synth, col=synth)) +
    geom_point(position=position_jitter(width=0.02,height=0.02))

plot(g)
