# Name: Santhosh Narayanan
# USC ID: 9197788157

from pyspark import SparkContext, SparkConf
import numpy as np
import time
from sklearn.cluster import KMeans
import sys
import random


def group_cluster_data(cls_idx):
    cls_data = {}
    for cls in range(len(cls_idx)):
        if cls_idx[cls] in cls_data:
            cls_data[cls_idx[cls]].append(clustr_data[cls])
        else:
            cls_data[cls_idx[cls]] = [clustr_data[cls]]
    return cls_data


def print_stats(ds_stats, cs_stats, retain_set, rounds, out_path):
    ds_points_count = 0
    cs_points_count = 0

    for k in ds_stats.keys():
        ds_points_count += ds_stats[k][0]

    for k in cs_stats.keys():
        cs_points_count += cs_stats[k][0]

    ret_set_size = len(retain_set)
    cs_cluster_count = len(cs_stats)

    w = open(out_path, "a+")
    w.write("Round " + str(rounds) + ": " + str(ds_points_count) + "," + str(cs_cluster_count) + "," +
            str(cs_points_count) + "," + str(ret_set_size) + "\n")
    # print(
    #     "Intermediate result: Round " + str(rounds) + ": " + str(ds_points_count) + "," + str(cs_cluster_count) + "," +
    #     str(cs_points_count) + "," + str(ret_set_size) + "\n")

    return


def update_maps(data_arr, data_id_dict, true_clstr_dict):
    for idx in range(len(data_arr)):
        # data to data id
        data_id_dict[str(np.array(data_arr[idx][2:]).astype(float))] = data_arr[idx][0]
        # data_id to cluster_id
        true_clstr_dict[data_arr[idx][0]] = data_arr[idx][1]
    return data_id_dict, true_clstr_dict


def merge_cs(cs_clustr1, min_cluster, cs_stats):
    data1 = cs_stats[cs_clustr1]
    data2 = cs_stats[min_cluster]

    # print("Main cluster : "+str(data1))
    # print("Target cluster : "+str(data2))

    new_n = data1[0] + data2[0]
    new_sum = data1[1] + data2[1]
    new_sum_sq = data1[2] + data2[2]
    data2[3].extend(data1[3])

    cs_stats[min_cluster] = [new_n, new_sum, new_sum_sq, data2[3]]
    del cs_stats[cs_clustr1]

    return cs_stats


def merge_cs_ds(cs_clustr, min_cluster, cs_stats, ds_stats, pred_clstr_dict):
    data1 = cs_stats[cs_clustr]
    data2 = ds_stats[min_cluster]

    new_n = data1[0] + data2[0]
    new_sum = data1[1] + data2[1]
    new_sum_sq = data1[2] + data2[2]

    point_id = data1[3]

    for id in point_id:
        pred_clstr_dict[str(id)] = str(min_cluster)

    ds_stats[min_cluster] = [new_n, new_sum, new_sum_sq]
    del cs_stats[cs_clustr]

    return cs_stats, ds_stats, pred_clstr_dict


if __name__ == '__main__':

    start = time.time()

    # Mapping data coordinates to data index
    data_id_dict = {}
    # Mapping data coordinates to clusters predicted
    pred_clstr_dict = {}
    # Mapping data coordinates to clusters in ground truth
    true_clstr_dict = {}
    # List for maintaining retain set
    retain_set = []
    # DS Statistics
    ds_stats = {}
    # CS Statistics
    cs_stats = {}

    # Taking input parameters
    in_path = sys.argv[1]
    n_cluster = int(sys.argv[2])
    out_path = sys.argv[3]

    f = open(out_path, "w")
    f.write("The intermediate results:\n")
    f.close()

    # Setting up spark context and streaming
    conf = SparkConf().setAppName("BFR").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # Ingesting the data as numpy array
    data = sc.textFile(in_path)
    data = data.map(lambda d: d.split(",")).collect()

    # sample length of 20% of the data and threshold for distance
    spl_len = int(len(data) * 0.2)
    st_idx = 0
    ed_idx = st_idx + spl_len

    threshold = 2 * np.sqrt(len(data[0][2:]))

    # print("Round: " + str(1)+"\n")
    # print("Start index: " + str(st_idx))
    # print("End index: " + str(ed_idx - 1))
    # st = time.time()
    # step 1: Loading 20% of the data randomly
    random.shuffle(data)
    data_arr = np.array(data[st_idx:ed_idx])
    clustr_data = data_arr[:, 2:].astype(float)
    # print("Current chunk shape: " + str(clustr_data.shape))
    # print("Step 1: " + str(time.time() - st) + '\n')

    data_id_dict, true_clstr_dict = update_maps(data_arr, data_id_dict, true_clstr_dict)

    # st = time.time()
    # step 2: Running K Means on the data loaded in step 1
    kmeans = KMeans(init='k-means++', n_clusters=5 * n_cluster, random_state=0).fit(clustr_data)
    cls_idx = kmeans.fit_predict(clustr_data)
    cls_data = group_cluster_data(cls_idx)

    # print("Step 2: " + str(time.time() - st) + '\n')

    # st = time.time()
    # step 3: Moving outliers from K Means to retain set (RS) and removing the outliers
    rs = 0
    for key in cls_data.keys():
        if len(cls_data[key]) == 1:
            rs += 1
            retain_set.append(cls_data[key][0])
            clustr_data = clustr_data[clustr_data != cls_data[key]].reshape(clustr_data.shape[0] - 1,
                                                                            clustr_data.shape[1])

    # print("Step 3: " + str(time.time() - st) + '\n')

    st = time.time()
    # step 4: Running K Means on the rest of the chunk (after removing the RS data points)
    kmeans = KMeans(init='k-means++', n_clusters=n_cluster, random_state=0).fit(clustr_data)
    cls_idx = kmeans.fit_predict(clustr_data)
    cls_data = group_cluster_data(cls_idx)

    # print("Step 4: " + str(time.time() - st) + '\n')
    ds = 0
    cs = 0
    # st = time.time()
    # step 5: Finding discard set statistics
    for clstr, c_data in cls_data.items():
        # Storing the predicted labels
        for d in cls_data[clstr]:
            index = data_id_dict[str(d)]
            pred_clstr_dict[index] = str(clstr)
        # Number of points, N
        N = len(c_data)
        ds += N
        arr_data = np.array(c_data)
        # Sum of points in d dimensions
        sum_d = np.sum(arr_data, axis=0)
        # Sum of squares of points in d dimensions
        sum_sq_d = np.sum(arr_data ** 2, axis=0)
        ds_stats[clstr] = [N, sum_d, sum_sq_d]

    # print("Step 5: " + str(time.time() - st) + '\n')

    st = time.time()
    # step 6: Run K Means on RS that can create CS and RS if, number of points in RS > 5 * number of clusters
    if len(retain_set) >= 5 * n_cluster:
        kmeans = KMeans(init='k-means++', n_clusters=5 * n_cluster, random_state=0).fit(retain_set)
        cls_idx = kmeans.fit_predict(retain_set)
        cls_data = group_cluster_data(cls_idx)
        # creating new retain set
        retain_set = []
        rs = 0
        for clstr in cls_data.keys():

            # if the cluster has only one point, it is retain set
            if len(cls_data[clstr]) == 1:
                rs += 1
                retain_set.append(cls_data[clstr][0])

            else:
                # Number of points, N
                cs += 1
                N = len(cls_data[clstr])
                # List for maintaining the data id in compression set
                point_id = []
                for point in cls_data[clstr]:
                    point_id.append(data_id_dict[str(point)])
                arr_data = np.array(cls_data[clstr])
                # Sum of points in d dimensions
                sum_d = np.sum(arr_data, axis=0)
                # Sum of squares of points in d dimensions
                sum_sq_d = np.sum(arr_data ** 2, axis=0)
                cs_stats[clstr] = [N, sum_d, sum_sq_d, point_id]

    # print("Step 6: " + str(time.time() - st) + '\n')

    # print("points assigned to cs: " + str(cs))
    # print("points assigned to ds: " + str(ds))
    # print("points assigned to rs: " + str(rs))
    # print("retain set after point assignment: " + str(len(retain_set)))

    print_stats(ds_stats, cs_stats, retain_set, 1, out_path)

    for rounds in range(2, 6):

        # step 7: Loading 20% more data

        st_idx = ed_idx
        ed_idx = ed_idx + spl_len

        # print("Round: " + str(rounds)+"\n")
        # print("Start index: " + str(st_idx))

        if rounds == 5:
            data_arr = np.array(data[st_idx:])
            # print("End index: " + str(len(data) - 1))
        else:
            data_arr = np.array(data[st_idx:ed_idx])
            # print("End index: " + str(ed_idx - 1))

        data_id_dict, true_clstr_dict = update_maps(data_arr, data_id_dict, true_clstr_dict)

        clustr_data = data_arr[:, 2:].astype(float)

        # print("Current chunk shape: " + str(clustr_data.shape))

        ds = 0
        rs = 0
        cs = 0

        for d_point in clustr_data:

            # step 8: For the new points, compare them to each of the DS using the Mahalanobis Distance and assign them
            # to the nearest DS clusters if the distance is < 2√𝑑

            min_distance = threshold
            min_cluster = -1

            for cluster in ds_stats.keys():

                N = ds_stats[cluster][0]
                d_sum = ds_stats[cluster][1]
                d_sumsq = ds_stats[cluster][2]

                centroid = d_sum / N
                std_dev = np.sqrt((d_sumsq / N) - np.square(d_sum / N))

                dist_sq = np.square((d_point - centroid) / std_dev)
                m_dist = np.sqrt(np.sum(dist_sq, 0))
                # print(m_dist)
                if m_dist < min_distance:
                    min_distance = m_dist
                    min_cluster = cluster

            if min_cluster != -1:
                ds += 1

                index = data_id_dict[str(d_point)]
                pred_clstr_dict[index] = str(min_cluster)

                old_data = ds_stats[min_cluster]
                new_n = old_data[0] + 1
                new_sum = old_data[1] + d_point
                new_sum_sq = old_data[2] + np.square(d_point)
                ds_stats[min_cluster] = [new_n, new_sum, new_sum_sq]

            else:
                # step 9: For the new points that are not assigned to DS clusters, using the Mahalanobis Distance and
                # assign the points to the nearest CS clusters if the distance is < 2√𝑑

                min_dist = threshold
                min_clstr = -1

                for cluster in cs_stats.keys():

                    N = cs_stats[cluster][0]
                    d_sum = cs_stats[cluster][1]
                    d_sumsq = cs_stats[cluster][2]

                    centroid = d_sum / N
                    std_dev = np.sqrt((d_sumsq / N) - np.square(d_sum / N))

                    dist_sq = np.square((d_point - centroid) / std_dev)
                    m_dist = np.sqrt(np.sum(dist_sq, 0))

                    if m_dist < min_dist:
                        min_distance = m_dist
                        min_cluster = cluster

                if min_clstr != -1:

                    cs += 1

                    index = data_id_dict[str(d_point)]

                    old_data = cs_stats[min_clstr]
                    new_n = old_data[0] + 1
                    new_sum = old_data[1] + d_point
                    new_sum_sq = old_data[2] + np.square(d_point)
                    old_data[3].append(index)

                    cs_stats[min_clstr] = [new_n, new_sum, new_sum_sq, old_data[3]]

                else:
                    rs += 1
                    # step 10: For the new points that are not assigned to a DS cluster or a CS cluster, assign them to
                    # RS.
                    retain_set.append(d_point)

        # print("points assigned to cs: " + str(cs))
        # print("points assigned to ds: " + str(ds))
        # print("points assigned to rs: " + str(rs))
        # print("retain set after point assignment: " + str(len(retain_set)))

        # step 11: Run K-Means on the RS with a large K (e.g., 5 times of the number of the input clusters) to generate
        # CS (clusters with more than one points) and RS (clusters with only one point).
        if len(retain_set) >= 5 * n_cluster:

            kmeans = KMeans(init='k-means++', n_clusters=5 * n_cluster, random_state=0).fit(retain_set)
            cls_idx = kmeans.fit_predict(retain_set)
            cls_data = group_cluster_data(cls_idx)

            # creating new retain set
            retain_set = []
            for clstr in cls_data.keys():

                # if the cluster has only one point, it is retain set
                if len(cls_data[clstr]) == 1:
                    retain_set.append(cls_data[clstr][0])

                else:
                    # print(cls_data[clstr])
                    # Number of points, N
                    N = len(cls_data[clstr])
                    arr_data = np.array(cls_data[clstr])
                    # Sum of points in d dimensions
                    sum_d = np.sum(arr_data, axis=0)
                    # Sum of squares of points in d dimensions
                    sum_sq_d = np.sum(arr_data ** 2, axis=0)
                    # Point id in cs
                    point_id = []
                    for point in cls_data[clstr]:
                        point_id.append(data_id_dict[str(point)])

                    newkey = 0
                    if clstr in cs_stats.keys():
                        while (newkey in cs_stats.keys()):
                            newkey = newkey + 1
                    else:
                        newkey = clstr

                    cs_stats[newkey] = [N, sum_d, sum_sq_d, point_id]

        # print("retain set after kmeans on rs: "+str(len(retain_set)))

        # step 12: Merge CS clusters that have a Mahalanobis Distance < 2√𝑑
        clustr_keys = list(cs_stats.keys())

        for cs_clustr1 in clustr_keys:

            min_dist = threshold
            min_cluster = cs_clustr1

            c1_N = cs_stats[cs_clustr1][0]
            c1_sum = cs_stats[cs_clustr1][1]
            c1_centroid = c1_sum / c1_N

            for cs_clustr2 in clustr_keys:

                if cs_clustr1 != cs_clustr2:
                    c2_N = cs_stats[cs_clustr2][0]
                    c2_sum = cs_stats[cs_clustr2][1]
                    c2_sum_sq = cs_stats[cs_clustr2][2]
                    c2_centroid = c2_sum / c2_N

                    c2_std_dev = np.sqrt((c2_sum_sq / c2_N) - np.square(c2_sum / c2_N))

                    dist_sq = np.square((c1_centroid - c2_centroid) / c2_std_dev)
                    m_dist = np.sqrt(np.sum(dist_sq, 0))

                    if m_dist < min_dist:
                        min_distance = m_dist
                        min_cluster = cs_clustr2

            if min_cluster != cs_clustr1:
                cs_stats = merge_cs(cs_clustr1, min_cluster, cs_stats)
                clustr_keys.remove(cs_clustr1)


        if rounds == 5:
            # step 13: merge CS clusters with DS clusters that have a Mahalanobis Distance < 2√𝑑

            clustr_keys = list(cs_stats.keys())

            for cs_clustr in clustr_keys:

                min_dist = threshold
                min_cluster = -1

                c1_N = cs_stats[cs_clustr][0]
                c1_sum = cs_stats[cs_clustr][1]
                c1_centroid = c1_sum / c1_N

                for ds_clustr in ds_stats.keys():

                    c2_N = ds_stats[ds_clustr][0]
                    c2_sum = ds_stats[ds_clustr][1]
                    c2_sum_sq = ds_stats[ds_clustr][2]
                    c2_centroid = c2_sum / c2_N

                    c2_std_dev = np.sqrt((c2_sum_sq / c2_N) - np.square(c2_sum / c2_N))

                    dist_sq = np.square((c1_centroid - c2_centroid) / c2_std_dev)
                    m_dist = np.sqrt(np.sum(dist_sq, 0))

                    if m_dist < min_dist:
                        min_distance = m_dist
                        min_cluster = ds_clustr

                if min_cluster != -1:
                    cs_stats, ds_stats, pred_clstr_dict = merge_cs_ds(cs_clustr, min_cluster, cs_stats, ds_stats,
                                                                      pred_clstr_dict)

        print_stats(ds_stats, cs_stats, retain_set, rounds, out_path)

    final_cluster_dict = {}

    for key, val in pred_clstr_dict.items():
        final_cluster_dict[int(key)] = int(val)

    for val in retain_set:
        final_cluster_dict[int(data_id_dict[str(val)])] = -1

    for val in cs_stats.values():
        points = val[3]
        for point in points:
            final_cluster_dict[int(point)] = -1

    s = open(out_path, "a+")

    s.write("\nThe clustering results:\n")

    for key in sorted(final_cluster_dict.keys()):
        s.write(str(key)+","+str(final_cluster_dict[key])+"\n")

    s.close()
    print(time.time() - start)
