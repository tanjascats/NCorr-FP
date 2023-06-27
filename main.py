from scheme import CategoricalNeighbourhood


def test_knn():
    scheme = CategoricalNeighbourhood(gamma=1)
    data = "datasets/adult.csv"
    fingerprinted_data = scheme.insertion(data, secret_key=123, recipient_id=4)

    suspect = scheme.detection(fingerprinted_data, secret_key=123, original_data=data)


if __name__ == '__main__':
    test_knn()
