import attacks.bit_flipping_attack
import datasets


def create_influential_cluster(gamma=4, k=300):
    attack = attacks.bit_flipping_attack.InfluentialRecordFlippingAttack()
    cluster = attack.find_influential_records(datasets.Adult(), 'all', gamma=gamma, k=k, sk=100, fp_len=128)
    cluster.to_csv('cluster_g{}_k{}_sk100.csv'.format(gamma, k), index=False)


create_influential_cluster(gamma=4, k=300)
