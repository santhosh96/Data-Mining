from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import binascii
import datetime
import sys

n = 1024
hs_val_count = 100
result = []


def find_trailing_zeroes(n):
    if n == 0:
        return 0
    count = 0
    while n & 1 == 0:
        n = n >> 1
        count += 1
    return count


def myhashs(user):
    params = [
        [9030, 9008, 14323], [4854, 14309, 14327], [11313, 856, 14341], [3864, 13519, 14347],
        [12467, 10487, 14369], [13092, 2970, 14387], [10636, 5667, 14389], [5795, 9204, 14401],
        [6584, 12092, 14407], [7004, 7248, 14411], [361, 6779, 14419], [13926, 10279, 14423],
        [3362, 195, 14431], [2778, 2709, 14437], [10173, 9151, 14447], [11045, 9742, 14449],
        [4913, 5712, 14461], [9336, 769, 14479], [8820, 7377, 14489], [4850, 10010, 14503],
        [2357, 7985, 14519], [3285, 3459, 14533], [2604, 14135, 14537], [3983, 13993, 14543],
        [930, 2306, 14549], [14307, 11849, 14551], [8629, 3897, 14557], [14400, 6175, 14561],
        [3120, 6641, 14563], [9261, 13549, 14591], [14117, 4017, 14593], [6793, 1433, 14621],
        [440, 14370, 14627], [6327, 12187, 14629], [8019, 685, 14633], [5776, 8153, 14639],
        [12916, 12907, 14653], [599, 8253, 14657], [11518, 8655, 14669], [12822, 2852, 14683],
        [5500, 14135, 14699], [1297, 9665, 14713], [11551, 13539, 14717], [9126, 7238, 14723],
        [3406, 7131, 14731], [111, 4921, 14737], [11283, 3805, 14741], [13133, 4482, 14747],
        [1578, 11296, 14753], [7916, 2457, 14759], [6530, 9759, 15973], [1994, 4445, 15991],
        [2344, 7945, 16001], [565, 6165, 16007], [14412, 8091, 16033], [1127, 748, 16057],
        [14533, 511, 16061], [10323, 3817, 16063], [10405, 1610, 16067], [4203, 12386, 16069],
        [2461, 7825, 16073], [2201, 9862, 16087], [3650, 7581, 16091], [5623, 5498, 16097],
        [13397, 12481, 16103], [1383, 11394, 16111], [3724, 10831, 16127], [9685, 9707, 16139],
        [8182, 8924, 16141], [1930, 13527, 16183], [502, 14666, 16187], [14271, 86, 16189],
        [9755, 9208, 16193], [6315, 14102, 16217], [7205, 6825, 16223], [2180, 407, 16229],
        [9366, 13241, 16231], [10122, 1345, 16249], [8270, 2301, 16253], [4880, 11518, 16267],
        [9119, 3828, 16273], [16094, 16089, 16301], [692, 11228, 16319], [3188, 14667, 16333],
        [5865, 8820, 16339], [10411, 14679, 16349], [12999, 3592, 16361], [4933, 11265, 16363],
        [6889, 6828, 16369], [9191, 15850, 16381], [7280, 12154, 16411], [13018, 13563, 16417],
        [12779, 802, 16421], [13888, 12369, 16427], [8953, 12689, 16433], [12156, 4200, 16447],
        [5440, 164, 16451], [6309, 385, 16453], [8258, 4729, 16477], [10758, 15738, 16481]
    ]
    enc_user = int(binascii.hexlify(user.encode('utf8')), 16)
    hs = [((p[0] * enc_user + p[1]) % p[2]) % n for p in params]
    return hs


def get_data(windowRDD, outpath):
    global hs_val_count
    global result
    # list for ground truth
    user_seen = []
    # list for hash_values
    hash_values = []
    for user in windowRDD.collect():
        user_seen.append(user)
        hash_values.append(myhashs(user))

    max_r_val = [0 for i in range(hs_val_count)]

    for h_ind in range(len(hash_values)):
        for r_ind in range(len(max_r_val)):
            max_r_val[r_ind] = max(max_r_val[r_ind], find_trailing_zeroes(hash_values[h_ind][r_ind]))

    max_r_val = [pow(2, r) for r in max_r_val]

    est_usr = float(sum(max_r_val))/float(len(max_r_val))
    f = open(outpath, "a")
    print("ratio : ", est_usr/len(set(user_seen)))
    out = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+","+str(len(set(user_seen)))+","+str(est_usr)+"\n"
    f.write(out)
    f.close()
    print(out)
    # print("Estimated number of distinct users : ", str(est_usr))
    # print("Actual number of distinct users : ", str(len(set(user_seen))), "\n")


if __name__ == '__main__':
    port = sys.argv[1]
    outpath = sys.argv[2]
    f = open(outpath, "w")
    f.write("Time,Ground Truth,Estimation\n")
    f.close()
    # setting up spark contex and streaming
    conf = SparkConf().setAppName("FlajoletMartin").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    # setting streaming context with batch duration as 5 seconds
    ssc = StreamingContext(sc, 5)
    lines = ssc.socketTextStream("localhost", int(port))
    window_data = lines.window(windowDuration=30, slideDuration=10)
    window_data.foreachRDD(lambda d: get_data(d, outpath))
    ssc.start()
    ssc.awaitTermination()