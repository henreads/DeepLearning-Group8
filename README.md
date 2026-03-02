# DeepLearning-Group8

# Automated IC Wafer Defect Classification  
**Group Number:** 08  
**Group Members:** 
- Henry Lee Jun, 1004219 
- Chia Tang, 1007200
- Genson Low, 1005931


---

## 1. Topic and Machine Learning Problem

### 1.1 Motivation

Semiconductor wafers form the foundation of modern critical infrastructure, including aerospace avionics, defence systems, medical devices, automotive control systems, and communication networks.  

Defects introduced during fabrication may arise from equipment drift, process contamination, thermal instability, lithographic misalignment, or other systematic process variations. If undetected, such defects can propagate downstream, resulting in yield loss, reliability degradation, and potentially catastrophic failures in safety-critical applications.

In real semiconductor fabrication environments, defective wafers are both rare and heterogeneous, and new failure modes may emerge over time without sufficient labelled examples. As a result, most wafers are normal, while defect samples are scarce and incomplete in coverage. Under such conditions, traditional supervised multi-class classification becomes impractical.

This motivates the adoption of an **unsupervised anomaly detection approach**, where models learn normal behaviour and detect deviations without requiring exhaustive defect labels.

---

### 1.2 Problem Statement

This project investigates **unsupervised anomaly detection** in semiconductor wafer manufacturing.

Instead of learning to classify predefined defect categories, the model will be trained primarily on wafers labelled as normal in order to learn the underlying distribution of healthy wafer patterns.

At inference time, the model must assign an **anomaly score** that reflects structural deviation from normal behaviour, without explicit supervision of all defect classes.

The objective is therefore:

- **Not** defect-type classification  
- **But** reliable identification of rare structural abnormalities under realistic class imbalance conditions  

---

## 2. Dataset Specification

This project utilises the **WM-811K (LSWMD)** wafer map dataset, which contains **811,457 wafer bin maps** represented as 2D matrices of die-level pass/fail information.

**Dataset Source:**  
[WM-811K (LSWMD) Wafer Map Dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map/code) 

---

### Dataset Distribution

| Category                         | Count     | Percentage |
|----------------------------------|-----------|------------|
| No-label                         | ~638,507  | 78.7%      |
| Labelled – None (Normal)         | 147,431   | 18.2%      |
| Labelled – Pattern (Defect)      | 25,519    | 3.1%       |
| **Total**                        | 811,457   | 100%       |

---

### Dataset Usage Strategy

Although 3.1% of wafers are explicitly labelled with defect patterns, 78.7% of samples are unlabeled and may contain undetected defects.

To avoid introducing label noise:

- The model will be trained using wafers explicitly labelled as **“None” (normal)**.
- Evaluation will be conducted on wafers with confirmed **defect patterns**.
- The unlabeled subset will be excluded from supervised evaluation to ensure reliability.

This setup preserves a **realistic rare-event scenario** while maintaining clean evaluation standards.
