pitch = c(233,204,242,130,112,142)
sex = c(rep("female",3),rep("male",3))
my.df = data.frame(sex,pitch)

xmdl = lm(pitch ~ sex, my.df)
summary(xmdl)

mean(my.df[my.df$sex=="female",]$pitch) # equals intercept

plot(fitted(xmdl),residuals(xmdl))
dfbeta(xmdl)




library('data.table')
impressions_all_ads = fread("./data/tables_impressions_all_ads_1.csv", header=F, stringsAsFactors = T)

dim(impressions_all_ads) # 2353065      42
# rm(impressions_all_ads)
