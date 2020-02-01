'''
Method Description:

Method: Model based collaborative filtering

Parameter values chosen for ALS train function:
rank = 2
iteration count = 15
lambda = 0.2

* During prediction, have taken the average value of predictions from the model and the average ratings of user and business from 
user.json and business.json files respectively.
* For handling the cold start problem, have taken the average value of user average rating and business average rating and a bias 
value of 2.5.

Error Distribution:
>=0 and <1: 97704
>=1 and <2: 37684
>=2 and <3: 6196
>=3 and <4: 458
>=4: 2

RMSE:
0.99533607823814

Execution Time:
69.91029405593872s
'''


from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import time
import json
import sys


def extract_json(data_path, col1, col2):
    '''
        Function for extratcing the json input file
    '''
    return sc.textFile(data_path).map(lambda d: json.loads(d)).map(lambda d: (d[col1], float(d[col2]))).collectAsMap()


def extract_csv(data_path):
    '''
        Function for extracting the csv input file
    '''
    raw = sc.textFile(data_path)
    header = raw.first()
    data = raw.filter(lambda d: d != header)
    return data.map(lambda d: d.split(",")).map(lambda d: (d[0], d[1], float(d[2])))


def extract_csv_test(data_path):
    '''
        Function for extracting the csv input file for test data
    '''
    raw = sc.textFile(data_path)
    header = raw.first()
    data = raw.filter(lambda d: d != header)
    return data.map(lambda d: d.split(",")).map(lambda d: (d[0], d[1]))


def get_id(data, pos):
    '''
        Function for extracting distinct value at pos position
    '''
    return data.map(lambda d: d[pos]).distinct()


def convert_map(data):
    '''
        Function for mapping the string id to integer id
    '''
    st_id = 0
    id_dict = {}
    rev_id_dict = {}
    for id in data:
        id_dict[id] = st_id
        rev_id_dict[st_id] = id
        st_id += 1 
    return id_dict, rev_id_dict


def model_based_cf(train_data, test_data, u_avg, b_avg, outpath):
    # universal set of user and business id
    univ_data = sc.union([train_data, test_data])
    user_id = get_id(univ_data, 0).collect()
    buss_id = get_id(univ_data, 1).collect()

    # mapping and reverse mapping for user and business id
    user_map, rev_user_map = convert_map(user_id)
    buss_map, rev_buss_map = convert_map(buss_id)

    # creating training and testing data
    train_ratings = train_data.map(lambda d: Rating(int(user_map[d[0]]), int(buss_map[d[1]]), float(d[2])))
    test_ratings = test_data.map(lambda d: ((int(user_map[d[0]]), int(buss_map[d[1]])), 1))
    test = test_ratings.map(lambda d: (d[0][0], d[0][1]))

    # training and prediction
    model = ALS.train(ratings=train_ratings, rank=2, iterations=15, lambda_=0.2)
    preds = model.predictAll(test).map(lambda r: ((r[0], r[1]), (r[2] + u_avg[rev_user_map[r[0]]] + b_avg[rev_buss_map[r[1]]]) / 3))

    # handling coldstart problem
    unpred = test_ratings.subtractByKey(preds)
    unpred_val = unpred.map(lambda d: ((d[0][0], d[0][1]), (b_avg[rev_buss_map[d[0][1]]] + u_avg[rev_user_map[d[0][0]]] + 2.5)/3))
    predict = sc.union([preds, unpred_val]) 
    # predictions = test_ratings.join(predict)

    # res = predictions.map(lambda r: abs(r[1][0] - r[1][1]))

    # err_less_1 = res.filter(lambda x: x >= 0 and x < 1).count()
    # err_less_2 = res.filter(lambda x: x >= 1 and x < 2).count()
    # err_less_3 = res.filter(lambda x: x >= 2 and x < 3).count()
    # err_less_4 = res.filter(lambda x: x >= 3 and x < 4).count()
    # err_gret_4 = res.filter(lambda x: x >= 4).count()

    # print("\nError Distribution:")
    # print(">=0 and <1: "+str(err_less_1))
    # print(">=1 and <2: "+str(err_less_2))
    # print(">=2 and <3: "+str(err_less_3))
    # print(">=3 and <4: "+str(err_less_4))
    # print(">=4: "+str(err_gret_4)+"\n")

    # mse = predictions.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    # print("RMSE: \n"+str(mse ** 0.5)+"\n")

    f = open(output_path, "w")
    f.write("user_id, business_id, prediction\n")

    for d in predict.collect():
        f.write(rev_user_map[d[0][0]] + ',' + rev_buss_map[d[0][1]] + ',' + str(d[1]) + '\n')

    f.close()


if __name__ == "__main__":
    
    start = time.time()
    conf = SparkConf().setAppName("collaborative_filtering").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # input format : data folder, test_file name, output_file name
    folder_name = sys.argv[1]
    test_data_path = sys.argv[2]
    output_path = sys.argv[3]

    train_f_name = "yelp_train.csv"
    user_f_name = "user.json"
    business_f_name = "business.json"

    train_data_path = folder_name + train_f_name
    user_data_path = folder_name + user_f_name
    buss_data_path = folder_name + business_f_name

    user_avg_rating = extract_json(user_data_path, "user_id", "average_stars")
    buss_avg_rating = extract_json(buss_data_path, "business_id", "stars")
    train_data = extract_csv(train_data_path)
    test_data = extract_csv_test(test_data_path)

    model_based_cf(train_data, test_data, user_avg_rating, buss_avg_rating, output_path)
    print("Execution Time:")
    print(str(time.time() - start)+"s")
    sc.stop()