from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import binascii
import datetime
import sys

# false positives
fp = 0
# true negatives
tn = 0
# list for maintaining the bit array
users_seen = []
# boolean list for maintaining global filter bit array
n = 69997
# bit array for bloom filter
bit_array = [False for i in range(n)]
# output result array
result = []


def myhashs(user):
    enc_user = int(binascii.hexlify(user.encode('utf8')), 16)
    params = [
        [9030, 9008, 14323], [4854, 14309, 14327],
        [11313, 856, 14341], [3864, 13519, 14347]
    ]
    hs = [((p[0] * enc_user + p[1]) % p[2]) % n for p in params]
    return hs


def check_existence(users, outpath):
    global fp
    global tn
    global result
    for user_id in users.collect():
        hash_list = myhashs(user_id)
        b_arr = True
        for pos in hash_list:
            b_arr = b_arr and bit_array[pos]
        # if the element is not found in the bit array, it is true negative
        if b_arr == False:
            tn += 1
        # it can be either true positive or a false positive
        elif user_id not in users_seen:
            fp += 1
    # print("false positive: " + str(fp) + " true negative :" + str(tn))
    fpr = 0
    f = open(outpath, "a")
    if fp == 0:
        out = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + "," + str(fp) + "\n"
    else:
        fpr = float(fp) / float(fp + tn)
        out = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + "," + str(fpr) + "\n"
    print(out)
    f.write(out)
    f.close()


def set_preprocess(users):
    for user_id in users.collect():
        # updating the user_seen set
        if user_id not in users_seen:
            users_seen.append(user_id)
        # updating the bit array
        hash_list = myhashs(user_id)
        for pos in hash_list:
            bit_array[pos] = True


if __name__ == '__main__':
    port = sys.argv[1]
    outpath = sys.argv[2]
    f = open(outpath, "w")
    f.write("Time,FPR\n")
    f.close()
    # setting up spark contex and streaming
    conf = SparkConf().setAppName("BloomFiltering").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    # setting streaming context with batch duration as 10 seconds
    ssc = StreamingContext(sc, 10)
    lines = ssc.socketTextStream("localhost", int(port))
    # checking the existence of the user using bit array and previously seen user list
    lines.foreachRDD(lambda d: check_existence(d, outpath))
    lines.foreachRDD(set_preprocess)
    ssc.start()
    ssc.awaitTermination()
