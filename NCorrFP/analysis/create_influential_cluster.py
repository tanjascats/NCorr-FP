import attacks.bit_flipping_attack
import datasets


def create_influential_cluster(gamma=1, k=325):
    attack = attacks.bit_flipping_attack.InfluentialRecordFlippingAttack()
    cluster = attack.find_influential_records(datasets.Adult(), 'all', gamma=gamma, k=k)
    cluster.to_csv('cluster_g{}_k{}_sk999.csv'.format(gamma, k), index=False)


create_influential_cluster(gamma=2, k=300)
