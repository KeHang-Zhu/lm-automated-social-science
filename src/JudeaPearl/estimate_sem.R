suppressPackageStartupMessages({
  library(lavaan)
  library(dplyr)}
)

args <- commandArgs(trailingOnly = TRUE)
directory <- args[1]
lavaan_syntax <- args[2]
standardized_estimates <- as.logical(args[3])

setwd(directory )

df <- read.csv('mapped_data.csv')

# Fit the model
fit <- sem(lavaan_syntax, data = df)
estimates <- parameterEstimates(fit, standardized = standardized_estimates)
estimates_df <-  as.data.frame(estimates)
print(estimates_df )
# Write the results to a file
write.csv(estimates_df, file = 'estimates_df.csv', row.names = FALSE)