from csv import DictReader

train = '../data/ad_counts_clicks.csv'  # path to training file

for t, row in enumerate(DictReader(open(train))):
    print(t, row)
import os
cwd = os.getcwd()
cwd
