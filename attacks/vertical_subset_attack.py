from attacks.attack import Attack
import time
import random


class VerticalSubsetAttack(Attack):

    def __init__(self):
        super().__init__()

    """
    Runs the attack; gets a subset of columns without (a) predefined column(s)
    """
    def run(self, dataset, columns):
        start = time.time()
        if 'Id' in columns:
            print("Cannot delete Id columns of the data set!")
            subset = None
        else:
            subset = dataset.drop(columns, axis=1)
            print("Vertical subset attack runtime on " + str(len(columns)) + " out of " + str(len(dataset.columns)-1) +
                  " columns: " + str(time.time()-start) + " sec.")
        return subset

    """
    Runs the attack; gets a subset of columns without random 'number_of_columns' columns
    Deletes 'numer_of_columns' columns
    """
    def run_random(self, dataset, number_of_columns, random_state):
        subset = dataset.copy()
        if number_of_columns >= len(dataset.columns)-1:
            print("Cannot delete all columns.")
            return None

        start = time.time()
        random.seed(random_state)
        column_subset = random.sample(list(dataset.columns.drop(labels=["Id"])), k=number_of_columns)
        subset = subset.drop(column_subset, axis=1)

        print("Vertical subset attack runtime on " + str(number_of_columns) + " (" + str(column_subset) + ") out of " +
              str(len(dataset.columns)-1) +
              " columns: " + str(time.time() - start) + " sec.")
        print('Released columns: ', subset.columns)
        return subset
