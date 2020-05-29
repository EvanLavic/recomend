# import pdfminer library for transform pdf in text
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

#import keras-bert and tesorflow librarys for work with bert model 
from keras_bert import extract_embeddings
import tensorflow as tf

import numpy as np #import numpy for comfortable work with data
import sqlite3 #import sqlite for make databases


#function for extract text from pdf, return full document text
def extract_text_from_pdf(pdf_name):
    resource_manager = PDFResourceManager()
    file_ha = io.StringIO()
    converter = TextConverter(resource_manager, file_ha)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    with open(pdf_name, "rb") as f:
        for page in PDFPage.get_pages(f, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = file_ha.getvalue()

    converter.close()
    file_ha.close()

    if text:
        return text

#vectorize function that transform text to vectors
def vectorize(text):
    model = "multi_cased_L-12_H-768_A-12"
    sentences = np.array(text.split("."))
    embeddings = np.array(extract_embeddings(model, sentences))

    #calculate means values to transform arrays in vector with 768 length
    text_vector = list()
    for sentence in embeddings:
        text_vector.append(np.mean(sentence, axis = 0))
    text_v = text_vector[0]
    for i in range(1,len(text_vector)):
        text_v += text_vector[i]
    return np.array(text_v/len(text_vector))

#function that solving some issues with tensorflow using in bert
def solving_tensorflow_issues():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def db_create():
    db = sqlite3.connect("books.db")
    cursor = db.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS vector_books 
                    (title text, vector text)""")
    return cursor, db

def db_insert(cursor, db, pdf_name, vector):
    str_vect = ""
    for feature in vector:
        str_vect += str(feature) + " "
    cursor.execute("""INSERT INTO vector_books values(?, ?)""", (pdf_name, str_vect))
    db.commit()


if __name__ == '__main__':
    cursor, db = db_create()
    # pdf_names = open("books_names.txt", "r").read().splitlines()
    pdf_names = ["09.04.02.pdf"]
    for pdf in pdf_names:
        ar_text = extract_text_from_pdf(pdf)
        solving_tensorflow_issues()
        vectors = vectorize(ar_text)
        db_insert(cursor, db, pdf, vectors)