# 1- Dataset characteristics

The table provides detailed statistical information about the datasets. The numbers in parentheses refer to the corresponding counts for the minority group. For example, in the WAL–AMZ dataset, \|D₁\| (2.6k) indicates that there are 2.6k entities, with 96 of them belonging to the minority group. The majority group parameters can be inferred from the table by subtracting the minority group numbers from the total values listed.


| Dataset    | #Attr. | \|D₁\|     | \|D₂\|      | \|P\|         | \|M\|      |
|------------|--------|------------|-------------|---------------|-----------|
| WAL–AMZ    | 5      | 2.6k (96)  | 22.0k (172) | 56.4m (2.5m)  | 962 (88)  |
| BEER       | 4      | 4.3k (1.3k)| 3.0k (932)  | 13.0m (6.8m)  | 68 (29)   |
| AMZ–GOO    | 3      | 1.4k (83)  | 3.2k (4)    | 4.4m (272.9k) | 1.2k (60) |
| FOD–ZAG    | 6      | 533 (72)   | 331 (63)    | 176.4k (25.2k)| 111 (10)  |
| ITU–AMZ    | 8      | 6.9k (1.9k)| 55.9k (12.7k)| 386.2m (171.0m)| 132 (40)  |
| DBLP–GOO   | 4      | 2.6k (191) | 64.3k (389) | 168.1m (13.2m)| 5.3k (403)|
| DBLP–ACM   | 4      | 2.6k (251) | 2.3k (225)  | 6.0m (1.1m)   | 2.2k (310)|

---
---


# 4- Assessing Bias Propagating from Blocking to Matching

This document presents the results of assessing bias propagation from blocking to matching using various blocking methods. The experiments focus on comparing fairness metrics such as Equal Opportunity (EO), Equalized Odds (EOP), and Demographic Parity (DP), along with confusion matrix elements (TP, FN, FP, TN) for minority and majority groups across several datasets.

## Blocking Methods

- **SB**: StandardBlocking
- **EQG**: ExtendedQGramsBlocking
- **ESA**: ExtendedSuffixArraysBlocking
- **QG**: QGramsBlocking
- **SA**: SuffixArraysBlocking
- **CTT**: CTT
- **AE**: AUTO

---

### Beer Dataset Results

#### StandardBlocking (SB)
- **EOP**: 0.0168
- **EO**: 0.0168
- **DP**: 1.7458e-06
- **Confusion Matrix**:
  - Minority: `TP=28, FN=1, FP=0, TN=6754455`
  - Majority: `TP=37, FN=2, FP=0, TN=6280477`

#### ExtendedQGramsBlocking (EQG)
- **EOP**: 0.0937
- **EO**: 0.0937
- **DP**: 1.2682e-06
- **Confusion Matrix**:
  - Minority: `TP=28, FN=1, FP=0, TN=6754455`
  - Majority: `TP=34, FN=5, FP=0, TN=6280477`

#### ExtendedSuffixArraysBlocking (ESA)
- **EOP**: 0.0592
- **EO**: 0.0592
- **DP**: 1.4162e-06
- **Confusion Matrix**:
  - Minority: `TP=27, FN=2, FP=0, TN=6754455`
  - Majority: `TP=34, FN=5, FP=0, TN=6280477`

#### QGramsBlocking (QG)
- **EOP**: 0.0681
- **EO**: 0.0681
- **DP**: 1.4274e-06
- **Confusion Matrix**:
  - Minority: `TP=28, FN=1, FP=0, TN=6754455`
  - Majority: `TP=35, FN=4, FP=0, TN=6280477`

#### SuffixArraysBlocking (SA)
- **EOP**: 0.0849
- **EO**: 0.0849
- **DP**: 1.2570e-06
- **Confusion Matrix**:
  - Minority: `TP=27, FN=2, FP=0, TN=6754455`
  - Majority: `TP=33, FN=6, FP=0, TN=6280477`

---

### Fodors-Zagat Dataset Results

#### StandardBlocking (SB)
- **EOP**: 0.0
- **EO**: 0.0
- **DP**: 0.0002317
- **Confusion Matrix**:
  - Minority: `TP=11, FN=0, FP=0, TN=25204`
  - Majority: `TP=101, FN=0, FP=0, TN=151107`

#### ExtendedQGramsBlocking (EQG)
- **EOP**: 0.0
- **EO**: 0.0
- **DP**: 0.0002317
- **Confusion Matrix**:
  - Minority: `TP=11, FN=0, FP=0, TN=25204`
  - Majority: `TP=101, FN=0, FP=0, TN=151107`

#### ExtendedSuffixArraysBlocking (ESA)
- **EOP**: 0.0297
- **EO**: 0.0297
- **DP**: 0.0002119
- **Confusion Matrix**:
  - Minority: `TP=11, FN=0, FP=0, TN=25204`
  - Majority: `TP=98, FN=3, FP=0, TN=151107`

#### QGramsBlocking (QG)
- **EOP**: 0.0
- **EO**: 0.0
- **DP**: 0.0002317
- **Confusion Matrix**:
  - Minority: `TP=11, FN=0, FP=0, TN=25204`
  - Majority: `TP=101, FN=0, FP=0, TN=151107`

#### SuffixArraysBlocking (SA)
- **EOP**: 0.0
- **EO**: 0.0
- **DP**: 0.0002317
- **Confusion Matrix**:
  - Minority: `TP=11, FN=0, FP=0, TN=25204`
  - Majority: `TP=101, FN=0, FP=0, TN=151107`

---

### Walmart-Amazon Dataset Results

#### StandardBlocking (SB)
- **EOP**: 0.0147
- **EO**: 0.0147
- **DP**: -1.7728e-05
- **Confusion Matrix**:
  - Minority: `TP=86, FN=2, FP=0, TN=2541792`
  - Majority: `TP=867, FN=7, FP=0, TN=53834242`

#### ExtendedQGramsBlocking (EQG)
- **EOP**: 0.0113
- **EO**: 0.0113
- **DP**: -1.7784e-05
- **Confusion Matrix**:
  - Minority: `TP=86, FN=2, FP=0, TN=2541792`
  - Majority: `TP=864, FN=10, FP=0, TN=53834242`

#### ExtendedSuffixArraysBlocking (ESA)
- **EOP**: 0.0106
- **EO**: 0.0106
- **DP**: -1.5915e-05
- **Confusion Matrix**:
  - Minority: `TP=77, FN=11, FP=0, TN=2541792`
  - Majority: `TP=774, FN=100, FP=0, TN=53834242`

#### QGramsBlocking (QG)
- **EOP**: 0.0147
- **EO**: 0.0147
- **DP**: -1.7728e-05
- **Confusion Matrix**:
  - Minority: `TP=86, FN=2, FP=0, TN=2541792`
  - Majority: `TP=867, FN=7, FP=0, TN=53834242`

#### SuffixArraysBlocking (SA)
- **EOP**: 0.0528
- **EO**: 0.0528
- **DP**: -1.5020e-05
- **Confusion Matrix**:
  - Minority: `TP=76, FN=12, FP=0, TN=2541792`
  - Majority: `TP=801, FN=73, FP=0, TN=53834242`

---

### Amazon-Google Dataset Results

#### StandardBlocking (SB)
- **EOP**: 0.0171
- **EO**: 0.0171
- **DP**: 5.1505e-05
- **Confusion Matrix**:
  - Minority: `TP=58, FN=2, FP=0, TN=272818`
  - Majority: `TP=1089, FN=18, FP=0, TN=4123053`

#### ExtendedQGramsBlocking (EQG)
- **EOP**: 0.0616
- **EO**: 0.0616
- **DP**: 5.9401e-05
- **Confusion Matrix**:
  - Minority: `TP=53, FN=7, FP=0, TN=272818`
  - Majority: `TP=1046, FN=61, FP=0, TN=4123053`

#### ExtendedSuffixArraysBlocking (ESA)
- **EOP**: 0.1816
- **EO**: 0.1816
- **DP**: 8.1097e-05
- **Confusion Matrix**:
  - Minority: `TP=40, FN=20, FP=0, TN=272818`
  - Majority: `TP=939, FN=168, FP=0, TN=4123053`

#### QGramsBlocking (QG)
- **EOP**: 0.0100
- **EO**: 0.0100
- **DP**: 4.4230e-05
- **Confusion Matrix**:
  - Minority: `TP=58, FN=2, FP=0, TN=272818`
  - Majority: `TP=1059, FN=48, FP=0, TN=4123053`

#### SuffixArraysBlocking (SA)
- **EOP**: 0.1601
- **EO**: 0.1601
- **DP**: 7.8562e-05
- **Confusion Matrix**:
  - Minority: `TP=44, FN=16, FP=0, TN=272818`
  - Majority: `TP=989, FN=118, FP=0, TN=4123053`

---



### DBLP-GoogleScholar Dataset Results

#### StandardBlocking (SB)
- **EOP**: 0.0065
- **EO**: 0.0065
- **DP**: 4.3342e-08
- **Confusion Matrix**:
  - Minority: `TP=37, FN=0, FP=0, TN=1128018`
  - Majority: `TP=460, FN=3, FP=0, TN=14005497`

#### ExtendedQGramsBlocking (EQG)
- **EOP**: 0.0108
- **EO**: 0.0108
- **DP**: -9.9454e-08
- **Confusion Matrix**:
  - Minority: `TP=37, FN=0, FP=0, TN=1128018`
  - Majority: `TP=458, FN=5, FP=0, TN=14005497`

#### ExtendedSuffixArraysBlocking (ESA)
- **EOP**: 0.0421
- **EO**: 0.0421
- **DP**: -1.1407e-06
- **Confusion Matrix**:
  - Minority: `TP=36, FN=1, FP=0, TN=1128018`
  - Majority: `TP=431, FN=32, FP=0, TN=14005497`

#### QGramsBlocking (QG)
- **EOP**: 0.0065
- **EO**: 0.0065
- **DP**: 4.3342e-08
- **Confusion Matrix**:
  - Minority: `TP=37, FN=0, FP=0, TN=1128018`
  - Majority: `TP=460, FN=3, FP=0, TN=14005497`

#### SuffixArraysBlocking (SA)
- **EOP**: 0.0518
- **EO**: 0.0518
- **DP**: -1.4560e-06
- **Confusion Matrix**:
  - Minority: `TP=37, FN=0, FP=0, TN=1128018`
  - Majority: `TP=439, FN=24, FP=0, TN=14005497`

---

### iTunes-Amazon Dataset Results

#### StandardBlocking (SB)
- **EOP**: 0.0
- **EO**: 0.0
- **DP**: 4.5370e-07
- **Confusion Matrix**:
  - Minority: `TP=2, FN=0, FP=0, TN=15804497`
  - Majority: `TP=11, FN=0, FP=0, TN=18957434`

#### ExtendedQGramsBlocking (EQG)
- **EOP**: 0.5
- **EO**: 0.5
- **DP**: 5.1697e-07
- **Confusion Matrix**:
  - Minority: `TP=1, FN=1, FP=0, TN=15804497`
  - Majority: `TP=11, FN=0, FP=0, TN=18957434`

#### ExtendedSuffixArraysBlocking (ESA)
- **EOP**: 0.2727
- **EO**: 0.2727
- **DP**: 2.9545e-07
- **Confusion Matrix**:
  - Minority: `TP=2, FN=0, FP=0, TN=15804497`
  - Majority: `TP=8, FN=3, FP=0, TN=18957434`

#### QGramsBlocking (QG)
- **EOP**: 0.0
- **EO**: 0.0
- **DP**: 4.5370e-07
- **Confusion Matrix**:
  - Minority: `TP=2, FN=0, FP=0, TN=15804497`
  - Majority: `TP=11, FN=0, FP=0, TN=18957434`

#### SuffixArraysBlocking (SA)
- **EOP**: 0.0
- **EO**: 0.0
- **DP**: 4.5370e-07
- **Confusion Matrix**:
  - Minority: `TP=2, FN=0, FP=0, TN=15804497`
  - Majority: `TP=11, FN=0, FP=0, TN=18957434`

---



The results demonstrate how different blocking methods affect bias metrics and classification performance for both minority and majority groups across various datasets.
