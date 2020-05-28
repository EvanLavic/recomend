import sqlite3
import numpy as np
from sklearn.cluster import MeanShift


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


if __name__ == '__main__':
    raw_user_data = open("names_and_marks.txt", "r").read().splitlines()
    split_data = np.array([line.split(";") for line in raw_user_data])
    names, marks = [], []
    for el in split_data:
        names.append(el[0])
        marks.append(int(el[1]))
    str_vectors = [db_selector(name) for name in names]
    usr_v = create_user_vector(str_vectors, marks)
    print (len(usr_v))
