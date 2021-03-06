library(synthpop)
a = synthpop::SD2011
vars_selection <- c("smoke", "sex", "age", "edu", "weight", 
                    "height", "bmi", "sport", "marital", "region", "wkabint", "income", "ls")
vars_classification <- c("smoke", "sex", "age", "edu", "sport", "marital", "wkabint", "ls")
a$wkabint <- as.character(a$wkabint)
a$wkabint[a$wkabint == "YES, TO EU COUNTRY" | a$wkabint == "YES, TO NON-EU COUNTRY"] <- "YES"
a$wkabint <- factor(a$wkabint)
#a$income[a$income == -8] <- NA

a = a[complete.cases(a[ , vars_classification]),]

write.csv(a[,vars_selection],
                       file="polish_data_2011.csv",
                       row.names=FALSE)

