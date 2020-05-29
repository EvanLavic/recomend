import sqlite3
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise

# function select all vectors wtih same book name
def db_selector(marked_book):
    db = sqlite3.connect("books.db")
    cursor = db.cursor()
    cursor.execute("""SELECT vector FROM vector_books WHERE title = (?)""", (marked_book,))
    vector = cursor.fetchall()
    return vector

#function that creating user vector by calculating mean value os book-vector summarization
def create_user_vector(str_vectors, marks):
    #transform str_vector in array of floats
    vectors = [el[0][0] for el in str_vectors]
    res = []
    for el in vectors:
        row = []
        for i in el.split():
            row.append(float(i))
        res.append(row)    
    res = np.array(res)

    #creating usr_vector
    N = 0.0
    usr_v = res[0] * marks[0] /5.0
    for i in range(1, len(res)):
        if marks[i]>=3:
            usr_v+=(res[i]*marks[i])/5.0
            N+=1
    return np.array(usr_v/N)

#function that clustering all books and calculate rec_vector, which include books_vector to recommend  
def clustering(usr_vec):
    #connecting to db 
    db = sqlite3.connect("books.db")
    cursor = db.cursor()
    cursor.execute("""SELECT vector FROM vector_books""")
    db_que = cursor.fetchall()
    
    #transform str_vector in array of floats
    vectors = []
    for vector in db_que:
        vectors.append([float(x) for x in vector[0].split()])
    
    #clustering by Kmeans method
    estimator = KMeans(random_state=0, n_clusters=1)
    estimator.fit(vectors)
    centers = estimator.cluster_centers_
    labels = estimator.labels_
    uniq_lab = np.unique(labels)
    
    #calculate distances by cosinus distances and choosing cluster
    distances = [pairwise.cosine_distances(usr_vec.reshape(1, -1), center.reshape(1, -1)) for center in centers]
    min_d = np.min(distances)
    index = distances.index(min_d)
    usr_clust_lab = uniq_lab[index]

    #searching for books in choosed cluster
    rec_vec = list()
    for i in range(len(labels)):
        if labels[i] == usr_clust_lab:
            rec_vec.append(vectors[i])
        return rec_vec

#function to selecting books titles by finded vector
def books_rec(rec, book_names):
    #transform vector in str form for using in searching in db
    rec_str = []
    for vec in rec:
        str_v = ""
        for el in vec:
            str_v += str(el) + " "
        rec_str.append(str_v)
    #select from db books title
    db = sqlite3.connect("books.db")
    cursor = db.cursor()
    db_que = list()
    for vec in rec_str:
        cursor.execute("""SELECT title FROM vector_books WHERE vector = (?)""", (vec,))
        db_que.append(cursor.fetchall())
    
    #choosing uniq books
    books = []
    for vector in db_que:
        books.append([x for x in vector[0]])
    rec_books = []
    for book in books:
        if book[0] not in book_names:
            print(book[0])
            rec_books.append(book[0])
    return rec_books


if __name__ == '__main__':
    raw_user_data = open("names_and_marks.txt", "r").read().splitlines()
    split_data = np.array([line.split(";") for line in raw_user_data])
    names, marks = [], []
    for el in split_data:
        names.append(el[0])
        marks.append(int(el[1]))
    str_vectors = [db_selector(name) for name in names]
    usr_v = create_user_vector(str_vectors, marks)
    rec_v = clustering(usr_v)
    books = books_rec(rec_v, names)
    print(len(books))
    with open("user_recomendation", "w") as f:
        if books:
            f.write(str(books))
