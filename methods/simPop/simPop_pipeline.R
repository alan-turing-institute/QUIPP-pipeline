### A first try at a synthesis pipeline using simPop ###
#
# simPop is a package that combines a number of methods to generate synthetic populations 
# from individual-level population samples and aggregated census data (microsimulation). Its intended use is 
# mostly for cases where spatial microsimulation is desirable, i.e. where the individuals belong to some
# hierarchical spatial structure like a household. Nevertheless, the functions available can be used for more
# general synthetic tasks without a need for this hierarchy to be present (although some of the function arguments
# related to households still need to be filled). The package provides implementations 
# of IPF, Simulated Annealing and model-based data synthesis methods for categorical, semi-continuous and continuous variables 
# (e.g. multinomial, logistic and two-step regression, tree-based methods) and some other methods which are useful 
# for creating household stuctures (alias sampling, a model-based method for cases where variable values 
# are correlated between household members, a method for synthesising sub-components of variables, a method to allocate 
# population is small areas).
# 
# The main steps to create a synthetic population data set are the following:
#   - Import the micro-data using specifyInput()
#   - Calibrate the micro-data using calibSample() - this implements IPF and requires some form of cross-tabulated census data
#     to calibrate against.
#   - Extend the data set to population size by first sampling some basic variables using simStructure()
#   - Synthesize more variables (categorical and continuous) using simCategorical() and simContinuous()
#   - Simulate variables components and/or allocate the population to small areas with simComponents() and simInitSpatial()
#   - Calibrate the (now fully synthesised) population micro-data using calibPop() - this used Simulated Annealing and
#     requires some form of cross-tabulated census data to calibrate against.
# 
# The above flow can be altered depending on needs and some of the functions can be used independently but with some care.
# For example:
#   - specifyInput + calibSample: Can be used to simply run IPF on any data set, regardless of whether the data set is a 
#      sample or a population. Initial weights need to be provided, as well as a household id variable. 
#   - specifyInput + simStructure + simCategorical + simContinuous: Can be used to simply synthesise data (with modele-based)
#     synthesis, without any calibration (IPF, etc). The size of the synthesised data set depends on the weights the initial data set
#     has been assigned (simStructure "replicates" individuals a number of times equal to their weights). If wieghts are all equal to 1.0
#     the synthesised data set will have the same size as the original. Note that synthesis is done once and not multiple times like in
#     multiple imputation algorithms.
# 
# The package has a number of settings and features, e.g. allows parallelisation for several functions.
#
# This simple pipeline reads the data and executs the full flow of simPop. The results are used to 
# calculate various utility metrics. The code section titles refer to the corresponding sections in the simPop paper.
# At the end of thee main pipeline, there are two small demos of alternative flows which do not use all of the
# steps but just focus either on IPF or on model-based simulation.



### Import libraries ###
list.of.packages <- c("simPop", "dplyr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(simPop)
library(dplyr)



### Set the seed ###
set.seed(1234)



### Utility functions ###
read_data <- function(dataset_name){
  
  if (dataset_name == "Austrian") {
    # Polish quality of life survey (2011)
    data("eusilcS", package = "simPop")
    data_full <- eusilcS
    
  } else if (dataset_name == "CensusUK2011"){
    # ONS Census 2011 teaching file
    data_full <- read.csv("../../datasets/rft-teaching-file/2011 Census Microdata Teaching File.csv",
                          header=TRUE, skip=1)
    
  } else stop("Unknown datsaset")
  
  return(data_full)
  
}



### Import data and set up objects ###

# embeded data set containing Austrian micro-data (Section 4.1) or UK census data
# CAUTION: UK data have not been tested yet with this pipeline
dataset_name <- "Austrian"
orig_data <- read_data(dataset_name)
nobs_original <- nrow(data_full)
print(paste("Data set '", dataset_name, "' read sucessfully. Number of rows: ", 
            nobs_original, sep=""))

# create main data storage object (dataObj) using specifyInput and store it to variable inp (Section 4.2)
# db030 and hsize are the household id and size in the data set. The size can be skipped but hhid is needed.
# db040 is the region (not necessary at this stage but throws error is later function if not there)
# rb050 are the initial weights (contained in the data set, compulsory to specify)
inp <- specifyInput(orig_data, hhid = "db030", hhsize = "hsize", 
                    strata = "db040", weight = "rb050")

# load embedded cross-tab data which contain frequencies by sex and region (Section 4.3)
data("totalsRG", package = "simPop")
data("totalsRGtab", package = "simPop")
print(totalsRGtab)



### Calibrate using IPF ###

# Adjust frequencies in cross-tab data to atrificially limit the size of the population and limit the amount of
# computation later (Section 4.3)
totalsRG$Freq <- totalsRG$Freq / 100
totalsRGtab <- totalsRGtab / 100

# Calibrate using IPF - calibSample (Section 4.3)
# IPF adjusts the initial weights of each individual in the micro-data to reflect the information coming from the
# cross-tab data. Two versions are shown which do the same thing. It assumes same variable naming in the two data sets.
weights.df <- calibSample(inp, totalsRG)
weights.tab <- calibSample(inp, totalsRGtab)
identical(weights.df, weights.tab)

# Add the weights to the inp object
addWeights(inp) <- calibSample(inp, totalsRGtab)

print("Sample calibration with IPF ran sucessfully")



### Extend the data set to population size ###

# Use simStructure to resample individuals + HH structure + some basic variables in order to reach the intended population size 
# (defined by the sum of weights) - Section 4.4. If the weights are all 1.0, the population size will be equal to 
# the original sample size.
# The basic variables should be as few as possible (e.g. one) to reduce the privacy risk. 
# The resampling retains realistic household structures based on the information contained in the original 
# (sample) data set.
# Note that this function creates a simPopObj object and stores it into variable synthP. This object contains info
# about the sample and population.
synthP <- simStructure(data = inp, 
                       method = "direct", 
                       basicHHvars = c("age", "rb090", "db040"))

# check what variables are in the synthP object
print(synthP)

print("Extension to population size ran sucessfully")



### Synthesise remaining variables ###

# Use simCategorical to synthesise all categorical variables of interest (Section 4.5).
# The function trains a model on the sample data (original) with the synthesised variable as the
# response and the other variables as predictors. It then uses the trained model to predict the
# probabilities of each category for each record in the population and draws from the distribution
# of probabilities.
# The user can choose which variables to synthesise and which models to use, e.g. multinomial, random
# forests, etc. It is also possible to choose which predictors to use for each variable and to take into
# account the relationships between household members wheen generating.
# Here, we synthesise economic status and citizenship using multinomial logistic regression.
synthP <- simCategorical(synthP, 
                         additional = c("pl030", "pb220a"), 
                         method = "multinom",
                         nr_cpus = 1)

# check what variables are in the synthP object
print(synthP)

print("Synthesis of categorical variables ran sucessfully")

# Use simContinuous to synthesise all continuous variables of interest (Section 4.6).
# There are two ways to do that:
# - Fitting a multinomial model and taking random draws from the intervals of the categories
#   into which predictions fall.
# - Apply a logistic regression to separate close-to-zero values from  the rest. Theen for non-zeros
#   apply a linear regression. Random noise is also added - either based on normal assumpation or
#   by sampling from the residuals.
# Here, net income is synthesised, using the two-step regression approach. After the function is called, the
# income values for people under the age of 16 are converted to NA.
synthP <- simContinuous(synthP, additional = "netIncome", method = "lm", nr_cpus = 1)
ageinc <- pop(synthP, var = c("age", "netIncome"))
ageinc$age <- as.numeric(as.character(ageinc$age))
ageinc[age < 16, netIncome := NA]
pop(synthP, var = "netIncome") <- ageinc$netIncome

# check what variables are in the synthP object
print(synthP)

print("Synthesis of continuous variables ran sucessfully")



### Simulation of variable components ###

# It is possible to simulate components that together add up to one variable, e.g. different
# sources of income that are combined to make up the total income. The inputs are the sample
# data containing all the components and the synthetic (population data) where only the
# combined variable is available. The output are the components in the population data. 
# Here this is applied to income after some pre-processing. The function simComponents is used.
# Section 4.7
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

# check what variables are in the synthP object
print(synthP)

print("Simulation of variable components ran sucessfully")



### Allocating the population to small areas ###

# Up to this stage the only spatial information is the strata (regions in this case)
# Using simInitSpatial we can simulate infromation on a finer level - e.g. districts.
# This requires a table that contains the known population of smaller areas for each
# larger area, either as number of persons or number of households. This is simulated here
# using the simulate_districts function and then the resulting table is passed to the 
# simIntiSpatial function.
# Section 4.8 - takes a few seconds to run.
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
head(table(census$district))

## ----Section 4.8. simSpat2-----------------------
tabHH <- as.data.frame(xtabs(rb050~ db040 + district, data = census[!duplicated(census$db030),]))
tabP <- as.data.frame(xtabs(rb050~ db040 + district, data = census))
colnames(tabP) <- colnames(tabHH) <- c("db040", "district", "Freq")

## ----Section 4.8. simSpat3-------------------------------
synthP <- simInitSpatial(synthP, additional = "district",
                         region = "db040", tspatialHH = tabHH, tspatialP = tabP, nr_cpus = 1)
head(popData(synthP), 2)

print("Allocation to smaller areas ran sucessfully")



### Calibrating the population using Simulated Annealing ###
# We can now calibrate the population we have synthesised previously using aggregated cross-tabulated
# data that usually come from censuses. Here, the aggregated data are synthesised.
# Section 4.9 - takes several minutes

# Synthesise (simulate) the census data that will be used for calibration
census <- simStructure(data = inp, method = "direct",
                       basicHHvars = c("age", "rb090", "db040"))
census <- simCategorical(census, 
                         additional = c("pl030", "pb220a"), 
                         method = "multinom",
                         nr_cpus = 1)
census <- data.frame(popData(census))

# create aggregated data (margins) for region, sex and economic status
margins <- as.data.frame(xtabs(~ db040 + rb090 + pl030, data = census))
margins$Freq <- as.numeric(margins$Freq)

# Embed aggregated data (margins) to synthP
synthP <- addKnownMargins(synthP, margins)

# Run simulated annealing on the population data set adjusting for the margins provided above.
synthPadj <- calibPop(synthP, split = "db040", temp = 1, 
                      eps.factor = 0.00005, maxiter = 200, 
                      temp.cooldown = 0.975, factor.cooldown = 0.85,
                      min.temp = 0.001, verbose = TRUE, nr_cpus = 1)

# compare frequencies of census and calibrated data sets to see if they are similar
# we expect small differences - close to zero for most entries
pop <- data.frame(popData(synthP))
popadj <- data.frame(popData(synthPadj))

tab.census <- ftable(census[, c("rb090", "db040", "pl030")])
tab_afterSA <- ftable(popadj[, c("rb090", "db040", "pl030")])
tab.census - tab_afterSA

print("Calibration of population with SA ran sucessfully")



### Utility metrics ###
# Here, we compare the synthesised data set (population) with the 
# original data set (sample) in various ways.
# The basis for comparison are the frquencies (counts) for various
# subsets of data. For the sample data, a weighted mean is computed
# instead of the simple counts in order to get estimates of the 
# expected population counts.

# Calculates the Horowitz-Thompson estimate of cunts (weighted mean)
dat <- data.frame(sampleData(synthP))
tableWt(dat$pl030, weights = dat$rb050)

# Metric 1: Mosaic plot of expected and realised frequencies (from sample and pop respectively)
# 3 variables are used 
# spTable performs the cross-tabulations for both sample and population
tab <- spTable(synthP, select = c("rb090", "db040", "hsize"))
spMosaic(tab, labeling = labeling_border(abbreviate = c(db040 = TRUE)))

# Metric 2: Second type of Mosaic plot
tab <- spTable(synthP, select = c("rb090", "pl030"))
spMosaic(tab, method = "color")

# Metric 3: CDFs for net income by sex
spCdfplot(synthP, "netIncome", cond = "rb090", layout = c(1, 2))

# CDFs of net income by region
spCdfplot(synthP, "netIncome", cond = "db040", layout = c(3, 3))

# Metric 4: Box plot for net income by sex
spBwplot(synthP, x = "netIncome", cond = "rb090", layout = c(1, 2))

# Metic 5: Model fitting coeefficients
# This fits a linear reegression to predict net income separately on the sample
# and population data sets and using all other variables as predictors.
# It plots the coefficients, showing which ones are close between the two cases.
myPlot <- function(lm1,lm2,scale=TRUE){
  s1 <- confint(lm1)
  p1 <- summary(lm1)$coefficients[, 1]
  sig1 <- as.logical(summary(lm1)$coef[, 4] < 0.05)
  s2 <- confint(lm2)
  p2 <- summary(lm2)$coefficients[, 1]
  sig2 <- summary(lm1)$coef[, 4] < 0.05
  ## scaled
  if(scale){
    p1 <- scale(p1, center = 270.7126, 760.8413)
    p2 <- scale(p2, center = 270.7126, 760.8413)
    s1 <- scale(s1, center = c(270.7126, 270.7126), c(760.8413, 760.8413))
    s2 <- scale(s2, center = c(270.7126, 270.7126), c(760.8413, 760.8413))
  }
  ## without intersept
  p1 <- p1[2:length(p1)]
  p2 <- p2[2:length(p2)]
  s1 <- s1[2:length(p1), ]
  s2 <- s2[2:length(p2), ]
  ylims <- c(min(c(s1, s2)), max(c(s1, s2)))
  par(mar = c(5,5,0,0))
  plot(x = (1:length(p1)), y = p1, pch = 1, 
       ylab = expression(beta[i]), xlab = "index of regression coefficients", 
       cex.lab = 1.4, type = "n", ylim = ylims)
  points(x = (1:length(p1))[!sig1], y = p1[!sig1], cex.lab = 1.4)
  points(x = (1:length(p1))[sig1], y = p1[sig1], cex.lab = 1.8, pch = 18)
  points(x = 1:length(p2)+0.2, y = p2, col = "gray", pch = 20)
  segments(x0 = 1:length(p1), x1 = 1:length(p1), y0 = s1[, 1], y1 = s1[, 2])
  abline(h = 0, col = "gray", lty = 1)
  legend("topleft", legend = c("original - significant", "original - non-significant", 
                               "simulated population"), lty = c(1,1,1),
         pch = c(18,1,20), col = c("black", "black", "gray"))
}
samp <- data.frame(sampleData((synthP)))
form <- formula("netIncome ~ age + rb090 + db040 + pb220a +pl030 + hsize+ py010n + py050n + py090n + py100n")
lm1 <- lm(form, data=samp, weights=samp$rb050)
pop <- data.frame(popData(synthP))
pop$age <- as.numeric(pop$age)
lm2 <- lm(form, data=pop)
myPlot(lm1, lm2, scale = FALSE)

print("Utility metrics ran sucessfully")



### Demo of simpler pipeline #1 ###
# This is a specifyInput + calibSample pipeline which shows how to simply run IPF on the input data
# and with the minimal amount of household-specific concepts/arguments. It also assumes all the weights 
# are 1.0 at the beginning.
dataset_name <- "Austrian"
orig_data <- read_data(dataset_name)
inp_p1 <- specifyInput(orig_data, hhid = "db030", weight = "rb050")
data("totalsRG", package = "simPop")
totalsRG$Freq <- totalsRG$Freq / 100
weights.df_p1 <- calibSample(inp_p1, totalsRG)
addWeights(inp_p1) <- calibSample(inp_p1, totalsRG)



### Demo of simpler pipeline #2 ###
# This is a specifyInput + simStructure + simCategorical pipeline which shows how to simply synthesise
# a dataset that has the same size as the input dataset without doing any calibration, just model-based
# synthesis and with the minimal amount of household-specific concepts/arguments. It sets all the weights 
# to 1.0 at the beginning to achieve the correct population size.
dataset_name <- "Austrian"
orig_data <- read_data(dataset_name)
orig_data$w1 <- 1.0
inp_p2 <- specifyInput(orig_data, hhid = "db030", weight = "w1", strata = "db040")
synthP_p2 <- simStructure(data = inp_p2, 
                       method = "direct", 
                       basicHHvars = c("age", "rb090", "db040"))
synthP_p2 <- simCategorical(synthP_p2, 
                         additional = c("pl030", "pb220a"), 
                         method = "multinom",
                         nr_cpus = 1)

