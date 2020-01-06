### Unit tests for synthpop pipeline ###

# source the utility functions file
#setwd("../")
source("util.R")

# install packages
#list.of.packages <- c("testthat") # dplyr for covariance
#new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
#if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")

# load testthat
library("testthat")

# test the data input function
test_that("Dimensions of inputted dfs are as expected", {
  expect_equal(dim(read_data("Polish")), c(5000, 35))
  expect_equal(dim(read_data("CensusUK2011")), c(569741, 18))
})