import sqlite3
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise

def db_selector(marked_books):
    db = sqlite3.connect("books.db")
    cursor = db.cursor()
    cursor.execute("""SELECT vector FROM vector_books WHERE title = (?)""", (marked_books,))
    vectors = cursor.fetchall()
    return vectors

def create_user_vector(str_vectors, marks):
    vectors = [el[0][0] for el in str_vectors]
    # vectors = [int(i) for el in vectors for i in el.split()]
    
    res = []
    for el in vectors:
        row = []
        for i in el.split():
            row.append(float(i))
        res.append(row)    
    res = np.array(res)
    usr_v = res[0] * marks[0] /5.0
    for i in range(1, len(res)):
        usr_v+=(res[i]*marks[i])
    return usr_v

def clustering(usr_vec):
    
    db = sqlite3.connect("books.db")
    cursor = db.cursor()
    cursor.execute("""SELECT vector FROM vector_books""")
    db_que = cursor.fetchall()
    # print(len(db_que))
    vectors = []
    for vector in db_que:
        # print(len(vector))
        vectors.append([float(x) for x in vector[0].split()])
    
    estimator = KMeans(random_state=0)
    estimator.fit(vectors)
    centers = estimator.cluster_centers_
    labels = estimator.labels_
    uniq_lab = np.unique(labels)
    
    distances = [pairwise.cosine_distances(usr_vec, center) for center in centers]
    min_d = np.min(distances)
    index = distances.index(min_d)
    usr_clust_lab = uniq_lab[index]

    recom = list()
    for i in range(len(labels)):
        if labels[i] == usr_clust_lab:
            recom.append(vectors[i])
    
    # distances = []
    # min_dis = []
    # for vector in vectors:
    #     distance = [pairwise.cosine_distances(vector, center) for center in centers]
    #     # distances.append(distance)
    #     min_dis = np.min(distance)
    #     ind_min = distance.index(min_dis)

    #     # distances.append([pairwise.cosine_distances(vector, center) for center in centers])
    # # min_d = np.min(distances)
    # indexes = [min_dis.tolist().index()]

    return vectors

if __name__ == '__main__':
    raw_user_data = open("names_and_marks.txt", "r").read().splitlines()
    split_data = np.array([line.split(";") for line in raw_user_data])
    names, marks = [], []
    for el in split_data:
        names.append(el[0])
        marks.append(int(el[1]))
    str_vectors = [db_selector(name) for name in names]
    # usr_v = create_user_vector(str_vectors, marks)
    vec = clustering()
    print(len(vec[0]))

    # print (len(usr_v))
