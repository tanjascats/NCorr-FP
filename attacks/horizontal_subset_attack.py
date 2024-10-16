from attacks.attack import Attack
import time


class HorizontalSubsetAttack(Attack):

    def __init__(self):
        super().__init__()

    """
    Runs the attack; gets a random subset of a dataset of size fraction*data_size
    fraction [0,1]
    """
    def run(self, dataset, fraction, random_state=None):
        if fraction < 0 or fraction > 1:
            return None

        start = time.time()
        subset = dataset.sample(frac=fraction, random_state=random_state)
        print("Subset attack runtime on " + str(int(fraction*len(dataset))) + " out of " + str(len(dataset)) +
              " entries: " + str(time.time()-start) + " sec.")
        return subset
