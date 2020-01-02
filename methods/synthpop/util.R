### Utility functions for synthpop ###

library(synthpop)

read_data <- function(dataset_name){
  
  if (dataset_name == "Polish") {
    # Polish quality of life survey (2011)
    data_full <- SD2011
    levels(data_full$edu) <- c("NONE", "VOC", "SEC", "HIGH")
    
  } else if (dataset_name == "CensusUK2011"){
    # ONS Census 2011 teaching file
    data_full <- read.csv("../../datasets/rft-teaching-file/2011 Census Microdata Teaching File.csv",
                          header=TRUE, skip=1)
    
  } else stop("Unknown datsaset")
  
  return(data_full)
  
}

