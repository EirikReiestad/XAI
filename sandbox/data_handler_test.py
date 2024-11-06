import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_handler import DataHandler

if __name__ == "__main__":
    positive_concepts = DataHandler()
    positive_concepts.load_data_from_path("box-not-exist.csv")
    negative_concepts = DataHandler()
    negative_concepts.load_data_from_path("random_negative.csv")
    positive_data, test_positive_data = positive_concepts.split(0.7)
    negative_data, test_negative_data = negative_concepts.split(0.7)
    pos_data, pos_labels = positive_data.get_data_lists()
    neg_data, neg_labels = negative_data.get_data_lists()
    pos_data = np.array(pos_data)
    neg_data = np.array(neg_data)
    pos_data = pos_data.reshape(pos_data.shape[0], -1)
    neg_data = neg_data.reshape(neg_data.shape[0], -1)
    pos_labels = np.ones(len(pos_data))
    neg_labels = np.zeros(len(neg_data))

    data = np.concatenate([pos_data, neg_data])
    labels = np.concatenate([pos_labels, neg_labels])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    regressor = LogisticRegression()
    regressor.fit(scaled_data, labels)

    test_data, _ = positive_data.get_data_lists()
    test_labels = np.ones(len(test_data))
    test_data = np.array(test_data)
    test_data = test_data.reshape(test_data.shape[0], -1)
    scaled_test_data = scaler.transform(test_data)

    score = regressor.score(scaled_test_data, test_labels)
    print(score)
