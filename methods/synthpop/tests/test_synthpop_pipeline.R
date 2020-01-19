### Unit tests for synthpop pipeline ###

# source the utility functions file
setwd("../")
source("util.R")

# install packages
#list.of.packages <- c("testthat") # dplyr for covariance
#new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
#if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")
library(synthpop)

# load testthat
library("testthat")

# test the data input function
test_that("Dimensions of input dfs are as expected", {
  expect_equal(dim(read_data("Polish")), c(5000, 35))
  expect_equal(dim(read_data("CensusUK2011")), c(569741, 18))
})

test_that("End-to-end pipeline for Polish ds returns reasonable output",
          {
            dataset_name <- "Polish"
            data_full <- read_data(dataset_name)
            nobs_original <- nrow(data_full)
            vars_selection <- c("smoke", "sex", "age", "edu", "weight", 
                                  "height", "bmi", "sport", "marital", "region")
            synthesis_methods <- c("sample", "logreg", "ctree", "polyreg", "ctree", 
                                     "ctree", "ctree", "ctree", "cart", "cart")
            vars_sequence <- c(1,2,3,4,8,9,10,5,6,7)
            rules <- list(marital = "age < 18 & sex == 'MALE'")
            rule_values <- list(marital = "SINGLE")
            proper <- FALSE
            population_inference = TRUE
            m <- 3
            k <- nobs_original
            type <- "binomial"
            seed <- 1234567
            smoothing = list("","","density","","density","density","","","","")
            names(smoothing) <- vars_selection
            data_original <- data_full[, vars_selection]
            # synthesis
            data_synth <- syn(data = data_original, 
                              visit.sequence = vars_sequence,
                              method = synthesis_methods,
                              rules = rules,
                              rvalues = rule_values,
                              m = m,
                              k = k,
                              seed = seed,
                              proper = proper,
                              smoothing = smoothing)
            # frequeencies comparison
            results <- compare(data_synth, data_original)
            discrepancies_list <- list()
            for (i in 1:length(results[[1]])) {
              
              for (j in 1:length(results[[1]][i])) {
                discrepancies_list <- append(discrepancies_list, (results[[1]][[i]][j][[1]][2,] 
                                                                  - results[[1]][[i]][j][[1]][1,]) 
                                             / results[[1]][[i]][j][[1]][1,])
              }
              
            }
            mean_discr <- mean(unlist(discrepancies_list, use.names=FALSE))
            ##model fit comparison
            #model <- glm.synds(formula = smoke ~ sex + age + edu + weight + sex * edu, 
            #                     data = data_synth, family = type)
            #result_2 <- compare(model, data_original)
            #mean_ci <- mean(unlist(result_2$ci.overlap))
            
            expect_that(mean_discr, is_less_than(0.05))
            #expect_that(mean_ci, is_more_than(0.6))
          }
          )

