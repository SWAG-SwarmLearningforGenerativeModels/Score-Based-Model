import csv
LABEL_FILE = "/home/patrick/datasets/vinDr-png/annotations/image_labels_train.csv"
IMG_BASE_PATH = "/home/patrick/datasets/vinDr-png/train_png/"

#read csv
with open(LABEL_FILE,newline="") as f:
    reader = csv.reader(f)
    csvData = [list(row) for row in reader]

head = csvData[0]
labelNames = head[2:]
sampleList = csvData[1:]

imgDict = dict()
for sample in sampleList:
    imgKey = sample[0]
    if imgKey not in imgDict:
        imgDict[imgKey] = list()
    imgDict[imgKey].append(sample)

datasetList = list()
getAvgVal = lambda lst: int((sum(lst)/len(lst)) + 0.5) # 1 if half of more of the values are 1, else 0

for key, val in imgDict.items():
    zipedLabel = list(zip(*val))[2:]
    imgList = [IMG_BASE_PATH + str(key) + ".png",]
    imgList += [getAvgVal([int(x) for x in tup]) for tup in zipedLabel]
    datasetList.append(imgList)





