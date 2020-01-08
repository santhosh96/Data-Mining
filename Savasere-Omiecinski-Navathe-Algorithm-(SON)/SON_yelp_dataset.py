from pyspark import SparkContext, SparkConf
from collections import OrderedDict
from itertools import combinations
from collections import OrderedDict
import sys
import time
import gc
from tqdm import tqdm

def findFrequents(candidateItems, chunk, pthreshold):
    
    frequents = []
    for item in candidateItems:
        itCount = 0
        for bucket in chunk:
            if set(item).issubset(bucket):
                itCount = itCount + 1
            if itCount >= pthreshold:
                frequents.append((item,1))
                break

    return frequents

def apriori(chunk, prevCandidates, pthreshold, itemSetSize):

    c_dict = {}

    candidateItems = []

    if len(prevCandidates) == 1:
        return candidateItems

    for i in range(len(prevCandidates)):
        for j in range(i+1,len(prevCandidates)):
            if itemSetSize == 2:
                candidateItems.append(tuple([prevCandidates[i][0][0],prevCandidates[j][0][0]]))
            else:
                s = sorted(list(set(prevCandidates[i][0]).union(set(prevCandidates[j][0]))))
                s = tuple(s)
                if len(s) == itemSetSize:
                    candidateItems.append(s) 

    candidateItems = list(set(candidateItems))

    return findFrequents(candidateItems, chunk, pthreshold)

def generateCandidates(chunk, threshold, buckets):

    # dictionary
    c_dict = {}
    chunk = list(chunk)
    
    # partition threshold (global threshold / number of chunks)
    pthreshold = (float(len(chunk)) / float(buckets)) * threshold

    # for itemset of size 1
    singleItemSet = set()
    for bucket in chunk:
        singleItemSet = singleItemSet.union(set(bucket))
    
    singleItemSet = list(map(lambda item : tuple([item]), list(singleItemSet)))

    intCandidates = findFrequents(singleItemSet, chunk, pthreshold)

    # candidate set
    candidates = []      

    # itemset size
    size = 2

    while(len(intCandidates) > 0):
        candidates.extend(sorted(intCandidates))
        intCandidates = apriori(chunk, sorted(intCandidates), pthreshold, size)
        size += 1

    return candidates

# function for generating the key value pair for itemset and the frequency
def generateFrequents(chunk, candidates):

    # dictionary for recording the (C, v)
    f_dict = {}

    for bucket in chunk:
        for item in candidates:    
            if set(item).issubset(bucket):
                if item in f_dict:
                    f_dict[item] = f_dict[item] + 1
                else:
                    f_dict[item] = 1  

    # list of (C, v) pair
    intFreq = []

    for k,v in f_dict.items():
        intFreq.append((k,v))

    return intFreq


# Program Starts here
if __name__ == "__main__":
    # spark-submit santhosh_narayanan_task1.py <case number> <support> <input_file_path> <output_file_path>

    # Creating the configuration for context with app name as yelpAnalysis and master URL as local
    conf = SparkConf().setAppName("sparkSON").setMaster("local[*]")
    # Creating the context object
    sc = SparkContext(conf=conf)
    # Setting log level to error
    sc.setLogLevel("ERROR")

    # case number for execution
    threshold = int(sys.argv[1])

    # support threshold
    support = int(sys.argv[2])

    # path of the dataset
    inpath = sys.argv[3]

    # path for the output
    outpath = sys.argv[4]

    start = time.time()

    # reading the dataset 
    raw = sc.textFile(inpath)
    # fetching the header of csv and removing the same
    header = raw.first()
    data = raw.filter(lambda data : data != header)

    # step 1
    data = data.map(lambda d : d.split(",")) \
                .map(lambda x : (x[0],x[1])) \
                .map(lambda x : (x[0],[x[1]])) \
                .reduceByKey(lambda b1, b2: b1 + b2).persist()

    # step 2
    data = data.filter(lambda u : len(u[1]) > threshold) \
            .map(lambda x : set(x[1]))

    buckets = data.count()

    # step 3
    # MapReduce for finding the candidate itemsets
    candidates = data.mapPartitions(lambda d : generateCandidates(d, support, buckets)) \
                    .reduceByKey(lambda x,y : 1) \
                    .sortByKey().map(lambda x : x[0])

    candidates = candidates.collect()

    # MapReduce for finding the frequent itemsets
    intFrequents = data.mapPartitions(lambda d : generateFrequents(d, candidates))
    frequents = intFrequents.reduceByKey(lambda x,y : x+y) \
                            .filter(lambda x : x[1] >= support) \
                            .sortByKey().map(lambda x : x[0]).persist()

    frequents = frequents.collect()

    candidateOutput = {}

    for c in candidates:
        size = len(c)
        if size in candidateOutput:
            candidateOutput[size].append(c)
        else:
            candidateOutput[size] = [c]

    frequentsOutput = {}

    for f in frequents:
        size = len(f)
        if size in frequentsOutput:
            frequentsOutput[size].append(f)
        else:
            frequentsOutput[size] = [f]

    f = open(outpath, "w")

    f.write("Candidates:\n")

    for k in range(1,len(candidateOutput)+1):
        line = ''
        for val in candidateOutput[k]:
            val = str(val)
            if k == 1:
                line += val.split(',')[0]+val.split(',')[1]+','
            else:
                line += val+','
        f.write(line[:-1]+'\n\n')

    f.write("Frequent Itemsets:\n")
    for k in range(1,len(frequentsOutput)+1):
        line = ''
        for val in frequentsOutput[k]:
            val = str(val)
            if k == 1:
                line += val.split(',')[0]+val.split(',')[1]+','
            else:
                line += val+','
        if k == len(frequentsOutput):
            f.write(line[:-1])
        else:
            f.write(line[:-1]+'\n\n')

    f.close()

    print("Duration: "+str(time.time()-start))
