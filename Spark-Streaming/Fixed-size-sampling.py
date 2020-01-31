from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import sys
import random
random.seed(553)

user_seq_num = {}
reservoir = []
seq_num = -1
reservoir_size = 100


def user_decision(seq_num, r_val):
    if r_val % seq_num < reservoir_size:
        return True
    return False


def find_user_sample(user_rdd, outpath):
    global user_seq_num
    global seq_num
    global reservoir
    global reservoir_size
    for usr in user_rdd.collect():
        seq_num += 1
        user_seq_num[usr] = seq_num
        # if the sequence number is less than 100, add it to reservoir
        if seq_num < reservoir_size:
            reservoir.append(usr)
        else:
            r_val = random.randint(0, 100000)
            # print(r_val)
            if (user_decision(seq_num, r_val)):
                ind = random.randint(0, 100000) % reservoir_size
                # print("random_value : "+str(r_val)+" usr_seq : "+str(seq_num)+" replace_index : "+str(ind))
                reservoir[ind] = usr
        if (seq_num + 1) % 100 == 0:
            print(len(user_seq_num))
            out = str(seq_num + 1) + "," + str(reservoir[0]) + "," + str(reservoir[20]) + "," + str(reservoir[40]) + \
                  "," + str(reservoir[60]) + "," + str(reservoir[80]) + "\n"
            f = open(outpath, "a")
            f.write(out)


if __name__ == '__main__':

    port = sys.argv[1]
    outpath = sys.argv[2]

    f = open(outpath, "w")
    f.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")
    f.close()
    # setting up spark context and streaming
    conf = SparkConf().setAppName("Reservoir Sampling").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # setting streaming context with batch duration as 10 seconds
    ssc = StreamingContext(sc, 10)
    lines = ssc.socketTextStream("localhost", int(port))

    lines.foreachRDD(lambda d: find_user_sample(d, outpath))
    ssc.start()
    ssc.awaitTermination()
