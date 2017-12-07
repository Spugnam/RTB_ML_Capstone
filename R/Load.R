####################################
################Sito################
####################################


####################
#Load
####################
library('data.table')
library('feather')

folder = "../data/split_by_day"

file_list <- list.files(path = folder,
                        all.files = FALSE, recursive = FALSE,
                        full.names = TRUE, # add file paths if true
                        pattern="*.csv$",
                        ignore.case = FALSE, include.dirs = FALSE, no.. = FALSE)

# Check file sizes
# info <- file.info(file_list)
# info$size_mb <- info$size/(1024 * 1024) 
# print(subset(info, select=c("size_mb")))

impressions = fread(file_list[1], header=F, stringsAsFactors = T)
row.num <- dim(impressions)[1]
for (file in file_list[-1]) {
  cat("Starting...", file, "\n")
  impressions.temp <- fread(file_list[1], header=F, stringsAsFactors = T)
  cat("adding", dim(impressions.temp)[1], "\n")
  impressions <- rbind(impressions, impressions.temp)
}

# Check final row count
dim(impressions)

# Manually write column names
colnames(impressions) <- c('adSize', 'adType', 'bestVenueName', 'deviceName', 'deviceType',
                           'gender', 'landingPage', 'os', 'region', 'venueType', 'timestamp_hour', 'timestamp_weekday',
                           'ad', 'campaign', 'day', 'clicked', 'TrainTestFlag', 'IAB1', 'IAB2', 'IAB3', 'IAB4',
                           'IAB5', 'IAB6', 'IAB7', 'IAB8', 'IAB9', 'IAB10', 'IAB11',
                           'IAB12', 'IAB13', 'IAB14', 'IAB15', 'IAB16', 'IAB17', 'IAB18',
                           'IAB19', 'IAB20', 'IAB21', 'IAB22', 'IAB23', 'IAB24', 'IAB25',
                           'IAB26')

# Check variables type
sapply(impressions, class)

# Change some of the numerical features to factor
impressions[, ad:=as.factor(ad)]
impressions[, campaign:=as.factor(campaign)]

# Save to RData file
save.folder = "../data/split_by_day/concat/impressions.feather"

# feather library
write_feather(impressions, save.folder)

# Free up memory
rm(impressions)





