from pyspark import SparkContext, SparkConf
import sys
import time

def count_co_rated(users):
    u_r = user_ratings.value
    s = set(u_r[users[0]]).intersection(u_r[users[1]])
    return (users, len(s))


def bfs(root):
    data = data_dict.value
    # dictionary for maintaining the level of each node
    level = {root: 0}
    # dictionary for maintaining the shortest path ways to reach a node, root will be reached by 1 way
    node_val = {root: 1}
    # inserting root into the stack
    queue = [root]
    # list for maintaining the traversal order
    traversal = [root]
    # dictionary for maintaining the parent nodes of current nodes
    parent_nodes = {root: []}

    # doing breadth-first search of the graph from the given root
    while queue:
        # current root is first in the queue
        parent = queue[0]
        # children of the root element
        children = data[parent]
        for child in children:
            # if the child is not recorded so far, new child is explored
            if child not in level:
                # append to traversal list
                traversal.append(child)
                # queue the node
                queue.append(child)
                # assign parent of the current child
                parent_nodes[child] = [parent]
                # assign shortest way to the child as parent value
                node_val[child] = node_val[parent]
                # assign level one more than the parent
                level[child] = level[parent]+1
            # if the child is recorded, but is not sibling or a parent
            elif child in level and level[child] == level[parent]+1:
                # add current parent to the list of child's parents list
                parent_nodes[child].extend([parent])
                # add shortest way value of the parent to the child (more than one shortest way to the current child)
                node_val[child] += node_val[parent]
        # remove the explored parent
        queue.pop(0)

    # dictionary for betweennes value for node and edge
    bt_node = {}
    bt_edge = {}

    # calculating betweenness value for the edge in reverse order of traversed nodes
    for node in reversed(traversal):
        # for new nodes, assign value of 1
        if node not in bt_node:
            bt_node[node] = 1
        # parents of the current node
        parents = [p for p in parent_nodes[node]]
        # shortest_path value for the parent
        parent_val = [node_val[p] for p in parents]
        # sum of the shortest path
        val_sum = sum(parent_val)
        for i, p in enumerate(parents):
            # finding the fraction of the value of the current node for the parent node(s)
            fraction = float(parent_val[i])/float(val_sum) * bt_node[node]
            # if the parent node does not exist, assign a value of 1
            if p not in bt_node:
                bt_node[p] = 1
            # assigning betweennes value for node and edge
            bt_node[p] = bt_node[p] + fraction
            # dividing the value by 2 for removing repetitiveness
            bt_edge[(min(p, node), max(p, node))] = fraction / 2

    return list(bt_edge.items())


def remove_edge(adj_list, betweenness):
    max_val = max(list(betweenness.values()))
    rem_edge = []
    for k in betweenness.keys():
        # print(betweenness[k])
        if betweenness[k] >= max_val:
            rem_edge.append(k)
    # print(rem_edge)
    for edge in rem_edge:
        l1 = adj_list[edge[0]]
        l1.remove(str(edge[1]))
        adj_list[edge[0]] = l1
        l2 = adj_list[edge[1]]
        l2.remove(str(edge[0]))
        adj_list[edge[1]] = l2
    return adj_list


def find_connected_components(adj_list):
    vertices = list(adj_list.keys())
    components = []
    while(vertices):
        queue = [vertices[0]]
        visited = []
        edges = []
        while(queue):
            root = queue[0]
            visited.append(root)
            vertices.remove(root)
            # print("root : ", root)
            # print("visited : ", visited)
            # print("vertices : ", vertices)
            for child in adj_list[root]:
                edges.append(tuple([min(root, child), max((root, child))]))
                if child not in visited and child not in queue:
                    queue.append(child)
            queue.pop(0)
        visited.sort()
        components.append(visited)

    return components


def find_modularity(components, m, k):
    m_val = 0
    for component in components:
        for i in component:
            for j in component:
                A_ij = 0
                if j in adjn_list[i]:
                    A_ij = 1
                k_i = k[i]
                k_j = k[j]
                m_val += A_ij - ((k_i * k_j)/(2 * m))
    m_val = m_val/(2*m)
    return m_val


if __name__ == "__main__":

    # usage: spark-submit task1.py video_small_num.csv output.csv

    start = time.time()

    conf = SparkConf().setAppName("GirwanNewman").setMaster("local[*]").set("spark.driver.host", "localhost")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    # in_path = 'video_small_num.csv'
    # out_path = 'output_task2.txt'

    raw = sc.textFile(in_path)
    header = raw.first()
    ratings = raw.filter(lambda d: d != header)

    ratings = ratings.map(lambda d: d.split(",")) \
        .map(lambda d: (d[0], d[1]))

    # finding the user and the products rated by the user
    user_ratings = ratings.map(lambda d: (d[0], [d[1]])) \
        .reduceByKey(lambda p1, p2: p1 + p2) \
        .sortByKey().collectAsMap()

    # broadcasting the dictionary across all the workers
    user_ratings = sc.broadcast(user_ratings)

    # users in the dataset
    users = ratings.map(lambda d: (d[0], 1)).distinct() \
        .sortByKey().map(lambda d: d[0])

    # finding the cartesian product of the users for finding the pairs of the users
    user_comb = users.cartesian(users).filter(lambda d: d[0] < d[1])

    # filtering out those users, who have number of commonly rated items less than 7
    user_rel = user_comb.map(count_co_rated).filter(lambda u: u[1] >= 7) \
                        .flatMap(lambda p: [(p[0][0], p[0][1]), (p[0][1], p[0][0])]) \
                        .map(lambda u: (u[0], [u[1]])).reduceByKey(lambda u1, u2: u1 + u2)

    data_dict = sc.broadcast(user_rel.collectAsMap())
    keys = user_rel.map(lambda d: d[0])
    adjn_list = user_rel.collectAsMap()

    betweenness = keys.flatMap(lambda k: bfs(k)).reduceByKey(lambda k1, k2: k1 + k2) \
                      .sortBy(lambda d: d[0][0]).sortBy(lambda d: -d[1]).collectAsMap()

    m = len(betweenness)
    k = {}
    for key, val in adjn_list.items():
        k[key] = len(val)
    final_components = []
    modularity_val = -1

    data_dict = remove_edge(data_dict.value, betweenness)
    components = find_connected_components(data_dict)
    modularity = round(find_modularity(components, m, k), 13)
    data_dict = sc.broadcast(data_dict)

    # print("Number of Components: ", len(components))
    # print("Current modularity value: ", modularity)
    if (modularity > modularity_val):
        modularity_val = modularity
        final_components = components
    # print("Maximum modularity value: ", modularity_val,"\n")

    while(True):

        betweenness = keys.flatMap(lambda k: bfs(k)).reduceByKey(lambda k1, k2: k1 + k2) \
            .sortBy(lambda d: d[0][0]).sortBy(lambda d: -d[1]).collectAsMap()

        data_dict = remove_edge(data_dict.value, betweenness)
        components = find_connected_components(data_dict)
        modularity = round(find_modularity(components, m, k), 13)
        data_dict = sc.broadcast(data_dict)

        print("current: ", modularity)
        print("max: ", modularity_val)

        if modularity >= modularity_val:
            modularity_val = modularity
            final_components = components
        else:
            break
        # print("Maximum modularity value: ", modularity_val,"\n")

    final_components.sort(key=lambda d: d[0])
    final_components.sort(key=lambda d: len(d))

    f = open(out_path, "w")
    for lis in final_components:
        f.write(str(lis)[1:-1])
        f.write('\n')

    print(time.time() - start)
