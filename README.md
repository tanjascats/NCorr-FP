# Data Fingerprinting for Tracing Unauthorised Usage

A fingerprint is a personalised, secret piece of information identifying both the data owner and the recipient of the data. By embedding the fingerprint into the data, the owner achieves two main goals: 
1. **tracing of the unauthorised data redistributor**
2. **ownership verification**

The fingerprint system is a two-stage process: 

![fingerprinting-system](https://github.com/tanjascats/NCorr-FP/blob/main/figures/fingerprinting-system.jpg)

The main properties of the fingerprinting system include:
- security - without access to the secret key, the fingerprint detection process cannot be performed correctly
- blindness - the fingerprint extraction does not require access to the original data
- public system - the fingerprinting system assumes that the method used for embedding the mark is public. Defence only lies in the choice of the private parameters (owner's secret key)
- robustness - the fingerprint can not be removed via benign data updates and malicious attacks without rendering the data useless
- utility - the fingerprint introduces only minor, insignificant modifications to the data


## NCorr-FP 
We improve the state-of-the-art [1-3] by developing a data-driven fingerprinting method for tabular data, **NCorr-FP** (Neighbourhood- and Correlation-based Fingerprinting). 
Building upon our earlier non-blind approach [4], the fingerprint is embedded by sampling the values from the existing distributions in the dataset, ensuring the high utility of the fingerprinted data while keeping the robustness of the fingerprint.

A fingerprint is an _L_-bit sequence where each bit determines how the new value will be sampled at a pseudo-random position in the dataset [5]. For each selected data value, there is a 50% chance the new value will be sampled from a low-density area of the value distribution in similar records and a 50% chance to be sampled from a high-density area of the value distribution in similar records. 
For example, below we show how a value distribution in similar records (the _neighbourhood_) might look like for a chosen data value. If the marking bit is 0 as depicted below, the new marked value is sampled from the low-density area (e.g. below 75th percentile). 

![demo-sampling](https://github.com/tanjascats/NCorr-FP/blob/main/figures/demo.png)

During the fingerprint detection, this process is reversed to decode the embedded bit. Hence, if the value falls in the low-density area, the embedded bit is assumed to be 0, otherwise 1. These extracted bit assumptions are added to the voting system. Each of the _L_ fingerprint bit gets assigned votes for the bit being 0 or 1. For _L_=16, the final voting might look like this:
![demo-votes](https://github.com/tanjascats/NCorr-FP/blob/main/figures/demo-votes.png)

The upper row represents the votes for bit 0 per bit-position, and the lower row represents the votes for bit 1 per bit-position. According to the majority vote, the fingerprint is decided to be the 16-bit sequence: 0100000001100100 which in a correct setup is the exact sequence associated to the recipient of the data copy. In reality, the fingerprint sequences are much longer (>100-bit) to ensure a small mutual overlap.

See the full demo at: [NCorrFP-demo-continuous.ipynb](https://github.com/tanjascats/NCorr-FP/blob/main/NCorrFP-demo-continuous.ipynb)


## Citation

If you use this code in your research, cite it as follows:
```
@misc{NCorr-FP,
  title={Neighbourhood- and Correlation-based Fingerprinting},
  author={Tanja Šarčević and Andreas Rauber and Rudolf Mayer},
  year={2024},
  url={https://github.com/tanjascats/NCorr-FP}
}
```

Author: Tanja Šarčević

## References: 
[1] Yilmaz, E. and Ayday, E., 2020. Collusion-resilient probabilistic fingerprinting scheme for correlated data. arXiv preprint arXiv:2001.09555.\
[2] Al Solami, E., Kamran, M., Saeed Alkatheiri, M., Rafiq, F. and Alghamdi, A.S., 2020. Fingerprinting of relational databases for stopping the data theft. Electronics, 9(7), p.1093.\
[3] Kieseberg, P., Schrittwieser, S., Mulazzani, M., Echizen, I. and Weippl, E., 2014. An algorithm for collusion-resistant anonymization and fingerprinting of sensitive microdata. Electronic Markets, 24, pp.113-124.\
[4] Sarcevic, T. and Mayer, R., 2020. A correlation-preserving fingerprinting technique for categorical data in relational databases. In ICT Systems Security and Privacy Protection: 35th IFIP TC 11 International Conference, SEC 2020, Maribor, Slovenia, September 21–23, 2020, Proceedings 35 (pp. 401-415). Springer International Publishing.\
[5] Li, Y., Swarup, V. and Jajodia, S., 2005. Fingerprinting relational databases: Schemes and specialties. IEEE Transactions on Dependable and Secure Computing, 2(1), pp.34-45.
