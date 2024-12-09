import numpy as np
import json
import re


def lsi_similarity(num_docs, documents, query, k):
    # Usunięcie interpunkcji i zbudowanie zbioru unikalnych słów
    clean_doc = lambda doc: re.sub(r'[^\w\s]', '', doc.lower())
    documents = [clean_doc(doc) for doc in documents]
    query = clean_doc(query)

    terms = sorted(set(word for doc in documents for word in doc.split()))

    # Budowa macierzy term-dokument (incydencji)
    term_doc_matrix = np.array([[1 if term in doc.split() else 0 for doc in documents] for term in terms])

    # Dekompozycja SVD
    U, Sigma, VT = np.linalg.svd(term_doc_matrix, full_matrices=False)

    # Redukcja wymiarów
    U_k = U[:, :k]
    Sigma_k = np.diag(Sigma[:k])
    VT_k = VT[:k, :]

    # Macierz dokumentów w zredukowanej przestrzeni
    reduced_docs = Sigma_k @ VT_k

    # Reprezentacja zapytania w zredukowanej przestrzeni
    query_vec = np.array([1 if term in query.split() else 0 for term in terms])
    reduced_query = np.linalg.pinv(Sigma_k) @ U_k.T @ query_vec

    # Obliczenie podobieństw kosinusowych
    similarities = [
        np.dot(reduced_query, doc_vec) / (np.linalg.norm(reduced_query) * np.linalg.norm(doc_vec))
        for doc_vec in reduced_docs.T
    ]

    # Zaokrąglenie wyników do 2 miejsc po przecinku
    similarities = [round(sim, 2) for sim in similarities]

    return similarities


if __name__ == "__main__":
    # Wczytanie danych wejściowych
    num_docs = int(input())
    documents = [input().strip() for _ in range(num_docs)]
    query = input().strip()
    k = int(input().strip())

    # Obliczenie podobieństw
    similarities = lsi_similarity(num_docs, documents, query, k)

    # Wyjście w formacie JSON
    print(json.dumps(similarities))
