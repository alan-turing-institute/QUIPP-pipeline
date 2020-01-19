### Unit tests for synthpop pipeline ###

# source the utility functions file
setwd("../")

# install packages
#list.of.packages <- c("testthat") # dplyr for covariance
#new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
#if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")
library(simPop)
library(dplyr)
library(tidyr)
library(magrittr)

# load testthat
library("testthat")


test_that("End-to-end pipeline for Austrian ds returns reasonable output",
          {
            
            
            
            
            # embeded data set containing Austrian micro-data (Section 4.1) or UK census data
            dataset_name <- "Austrian"
            data("eusilcS", package = "simPop")
            orig_data <- eusilcS
            nobs_original <- nrow(orig_data)
            inp <- specifyInput(orig_data, hhid = "db030", hhsize = "hsize", 
                                  strata = "db040", weight = "rb050")
            data("totalsRG", package = "simPop")
            orig_data_agg <- totalsRG
            orig_data_agg$Freq <- orig_data_agg$Freq / 100
            
            # Calibrate using IPF - calibSample (Section 4.3)
            weights.df <- calibSample(inp, orig_data_agg)
            
            # Add the weights to the inp object. 
            addWeights(inp) <- calibSample(inp, orig_data_agg)
            
            # Use simStructure to resample individuals + HH structure + some basic variables in order to reach the intended population size 
            synthP <- simStructure(data = inp, 
                                     method = "direct", 
                                     basicHHvars = c("age", "rb090", "db040"))
            
            # Use simCategorical to synthesise all categorical variables of interest (Section 4.5).
            synthP <- simCategorical(synthP, 
                                       additional = c("pl030", "pb220a"), 
                                       method = "multinom",
                                       nr_cpus = 1)
            # Use simContinuous to synthesise all continuous variables of interest (Section 4.6).
            synthP <- simContinuous(synthP, additional = "netIncome", method = "lm", nr_cpus = 1)
            ageinc <- pop(synthP, var = c("age", "netIncome"))
            ageinc$age <- as.numeric(as.character(ageinc$age))
            ageinc[age < 16, netIncome := NA]
            pop(synthP, var = "netIncome") <- ageinc$netIncome
            
            sIncome <- manageSimPopObj(synthP, var = "netIncome", sample = TRUE)
            sWeight <- manageSimPopObj(synthP, var = "rb050", sample = TRUE)
            pIncome <- manageSimPopObj(synthP, var = "netIncome")
            breaks <- getBreaks(x = sIncome, w = sWeight, upper = Inf, 
                                equidist = FALSE)
            synthP <- manageSimPopObj(synthP, var = "netIncomeCat", 
                                      sample = TRUE, set = TRUE, 
                                      values = getCat(x = sIncome, breaks))
            synthP <- manageSimPopObj(synthP, var = "netIncomeCat", 
                                      sample = FALSE, set = TRUE,
                                      values = getCat(x = pIncome, breaks))
            synthP <- simComponents(simPopObj = synthP, total = "netIncome",
                                    components = c("py010n", "py050n", "py090n", 
                                                   "py100n", "py110n", "py120n", "py130n", "py140n"), 
                                    conditional = c("netIncomeCat", "pl030"),
                                    replaceEmpty = "sequential", seed = 1)
            
            simulate_districts <- function(inp) {
                hhid <- "db030"
                region <- "db040"
                a <- inp[!duplicated(inp[, hhid]), c(hhid, region)]
                spl <- split(a, a[, region])
                regions <- unique(inp[, region])
                
                tmpres <- lapply(1:length(spl), function(x) {
                  codes <- paste(x, 1:sample(10:90, 1), sep = "")
                  spl[[x]]$district <- sample(codes, nrow(spl[[x]]), replace = TRUE)
                  spl[[x]]
                })
                tmpres <- do.call("rbind", tmpres)
                tmpres <- tmpres[, -2]
                out <- merge(inp, tmpres, by.x = hhid, by.y = hhid, all.x = TRUE)
                invisible(out)
            }
            data("eusilcS", package = "simPop")
            census <- simulate_districts(eusilcS)
            
            tabHH <- as.data.frame(xtabs(rb050~ db040 + district, data = census[!duplicated(census$db030),]))
            tabP <- as.data.frame(xtabs(rb050~ db040 + district, data = census))
            colnames(tabP) <- colnames(tabHH) <- c("db040", "district", "Freq")
            
            synthP <- simInitSpatial(synthP, additional = "district",
                                       region = "db040", tspatialHH = tabHH, tspatialP = tabP, nr_cpus = 1)
            
            census <- simStructure(data = inp, method = "direct",
                                     basicHHvars = c("age", "rb090", "db040"))
            census <- simCategorical(census, 
                                       additional = c("pl030", "pb220a"), 
                                       method = "multinom",
                                       nr_cpus = 1)
            census <- data.frame(popData(census))
              
            margins <- as.data.frame(xtabs(~ db040 + rb090 + pl030, data = census))
            margins$Freq <- as.numeric(margins$Freq)
            
            synthP <- addKnownMargins(synthP, margins)
            
            synthPadj <- calibPop(synthP, split = "db040", temp = 1, 
                                    eps.factor = 0.00005, maxiter = 200, 
                                    temp.cooldown = 0.975, factor.cooldown = 0.85,
                                    min.temp = 0.001, verbose = TRUE, nr_cpus = 1)
              
            pop <- data.frame(popData(synthP))
            popadj <- data.frame(popData(synthPadj))
              
            tab.census <- ftable(census[, c("rb090", "db040", "pl030")])
            tab_afterSA <- ftable(popadj[, c("rb090", "db040", "pl030")])
            diff <- mean(abs(tab.census - tab_afterSA) /tab.census, na.rm = TRUE)
            
            expect_lt(diff, 0.01)
            
          }
)

