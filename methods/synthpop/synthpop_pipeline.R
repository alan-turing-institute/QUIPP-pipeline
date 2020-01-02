### A first try at a synthesis pipeline using synthpop ###
#
# synthpop implements a version of multiple imputations (although the authors
# do not characterise it as such in the documentation and papers). The aim of
# the package is to create synthetic data sets in situations where the 
# original data set cannot be released for privacy or other reasons. 
#
# The approach followed is to sample from the joint distribution of the sensitive variables
# (given the known variables and the unknown parameters of the joint distribution).
# Because the parameters are unknown, the first step is to infer them by 
# fitting a model to the original data set (the type of model can be adapted by 
# the user). The fitting is done on the conditional 
# distirbutions of each variable in sequence. After fitting each conditional, samples
# are taken from it and placed in the synthetic data set. The process is repeated 
# for all variables to complete the synthetic data set. The process is then repeated m times
# to generate multiple synthetic data sets and then they can all be used to perform any
# inference task the user is interested about and their results combined.
#
# The package allows to skip variables, handle missing data, apply rules to variables, 
# some ways to tune privacy and utility.
# This simple pipeline reads data from a file, sets parameters for the package (including privacy settings) 
# and then runs the synthesis. The results are used to calculate various utility metrics.
# The utility is measured by examining descriptive statistics of the synthesised data sets vs the 
# original and also by fitting a model and comparing the inferences to the ones from the original.
#
# At  the moment the pipeline can use the Polish data set embedded in the synthpop package OR the ONS Census data set
# available in the repository.



### Import libraries ###
list.of.packages <- c("synthpop") # dplyr for covariance
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")

library(synthpop)
#library(dplyr)



### Utility functions ###
source("util.R")



### Read the data ###

# Datasets available: "Polish", "CensusUK2011"
dataset_name <- "CensusUK2011"
data_full <- read_data(dataset_name)
nobs_original <- nrow(data_full)
print(paste("Data set '", dataset_name, "' read sucessfully. Number of rows: ", 
            nobs_original, sep=""))


### Set parameters ###

# which variables do you want to use? these need to be consistent with the df cols
if (dataset_name == "Polish"){
  vars_selection <- c("smoke", "sex", "age", "edu", "weight", 
                      "height", "bmi", "sport", "marital", "region")
} else if (dataset_name == "CensusUK2011"){
  vars_selection <- c("Health", "Sex", "Age", "Marital.Status", "Student")
}

# which method to fit each variable? see docs for options
if (dataset_name == "Polish"){
  synthesis_methods <- c("sample", "logreg", "ctree", "polyreg", "ctree", 
                         "ctree", "ctree", "ctree", "cart", "cart")
} else if (dataset_name == "CensusUK2011"){
  synthesis_methods <- c("sample", "ctree", "ctree", "ctree", "cart")
}

# in what sequence should the variables be synthesised?
if (dataset_name == "Polish"){
  vars_sequence <- c(1,2,3,4,8,9,10,5,6,7)
} else if (dataset_name == "CensusUK2011"){
  vars_sequence <- c(1,2,3,4,5)
}



# definition of any rules/restrictions on values of variables
if (dataset_name == "Polish"){
  rules <- list(marital = "age < 18 & sex == 'MALE'")
  rule_values <- list(marital = "SINGLE")
} else if (dataset_name == "CensusUK2011"){
  rules <- list(Marital.Status = "Age == 1")
  rule_values <- list(Marital.Status = 1)
}

# Proper synthesis? The default is FALSE
# in which case synthetic data are generated from fitted 
# conditional distributions using the MLE estimate of 
# parameters. If TRUE they are generated using the whole
# posterior. When TRUE it is expected the utility
# metrics will improve (this needs to be investigated)
proper <- FALSE

# Compare utility for synthetic data vs original for population inference at the end? 
# If FALSE, utility is compared for original data set inference. 
population_inference = TRUE

# Number of independent synthesised data sets
m <- 3

# Number of observations per synthesised data set
k <- nobs_original

# NOT WORKING - TO BE FIXED: GLM type and formula - for utility quantification
# For some reason the synthpop compare() function throws an error when the formula
# variable is passed to the glm fit function.
#formula <- "smoke ~ sex + age + edu + weight + sex * edu"
type <- "binomial"

# random seed
seed <- 1234567

# privacy settings
# tree_minbucket: Defines the minimum number of individuals in tree leaves
# (for CTREE only) - the larger the number the more privacy is 
# maintained as it is more difficult to replicate real persons. At the same
# time the quality of the fit drops and thus utility drops.
# smoothing: Smoothing prevents releasing real unusual values for continuous variables
# (for ctree, cart, normrank, nested and sample methods).
# Options are "" or "density" and the argument needs to be a list - one element
# for each variable
tree_minbucket <- 5
if (dataset_name == "Polish"){
  smoothing = list("","","density","","density","density","","","","")
} else if (dataset_name == "CensusUK2011"){
  smoothing = list("","","","","")
}
names(smoothing) <- vars_selection

# one of the necessary checks for correct input parameters
if ((length(vars_selection)==length(synthesis_methods)) &
    (length(vars_selection)==length(vars_sequence)) &
    (length(vars_selection)==length(smoothing))){
  print("Parameters set sucessfully")
} else {
  stop("Parameter dimensions are inconsistent")
}




### Run the synthesis ###
data_original <- data_full[, vars_selection]
data_synth <- syn(data = data_original, 
                  visit.sequence = vars_sequence,
                  method = synthesis_methods,
                  rules = rules,
                  rvalues = rule_values,
                  m = m,
                  k = k,
                  seed = seed,
                  proper = proper,
                  ctree.minbucket = tree_minbucket,
                  smoothing = smoothing)
print("Synthesis ran sucessfully")



### Utility metrics ###

# Metric 1: Compare frequency distributions of original vs. synthesised for each variable
# Produces tables and graphs.
# Comment: The tables could be combined in various ways to form a single utility metric
print("Utility metric 1: Frequencies")
compare(data_synth, data_original)

# Metric 2: Compare various statistics of original vs. synthesised for each variable
# Again, these could be unified to single or a few numbers that summarise similarity
# Various statistical test could be added to confirm similarity 
# Note: Only the first of the m synthesised data sets is used here
print("Utility metric 2: Desriptive stats")
summary(data_original)
summary(data_synth$syn[[1]])

# Metric 3: Compare covariance matrices - again only m=1 is used for synthetic
# Uses only rows with no missing data
print("Utility metric 3: Covariance matrices")
#cov(dplyr::select_if(data_original, is.numeric), use = "complete.obs")
#cov(dplyr::select_if(data_synth$syn[[1]], is.numeric), use = "complete.obs")

# Metric 4: Train binomial GLM with interactions on synthetic data set, show results.
# Also, compare results with the same GLM trained on the original data set using the 
# embedded compare() function  - provides several utility metrics and graphs, see paper
# for details.
# Note: Set population.inference = TRUE in summary to capture coefficients estimates 
# and std. errors when inferring population coefficients. Set to  FALSE when  inferring
# the coefficient  estimates you  would get from the original sample.
print("Utility metric 4: GLM inference")
if (dataset_name == "Polish"){
  model <- glm.synds(formula = smoke ~ sex + age + edu + weight + sex * edu, 
                     data = data_synth, family = type)
} else if (dataset_name == "CensusUK2011"){
  
  model <- glm.synds(formula = Student - 1 ~ Sex + Age + Marital.Status + Health, 
                     data = data_synth, family = type)
}

summary(model, population.inference = population_inference)
compare(model, data_original)
print("Utility metrics ran sucessfully")



### Privacy metrics ###
# TBD...



