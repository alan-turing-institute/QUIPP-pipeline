### Unit tests for synthpop pipeline ###

# source the utility functions file
#setwd("../")
#source("mice_example.R")



# load testthat
library("testthat")
library(mvtnorm)
library(mice)
library(ggplot2)

Nsample = 20
Nsynth = 100

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
samples_imp = mice(samples, method="norm.boot", maxit=50)

samples_completed = complete(samples_imp)

## Is this row synthetic? Useful for plotting
samples_completed[1:Nsample, "synth"] = FALSE
samples_completed[(1 + Nsample):(Nsample + Nsynth), "synth"] = TRUE


# test the data input function
test_that("Dimension of output is as expected", {
  expect_equal(dim(samples_completed), c(Nsample + Nsynth, 3))
})

# failing test
test_that("fails", {
  expect_equal(1, 0)
})