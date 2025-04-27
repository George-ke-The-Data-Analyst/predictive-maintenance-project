# Predictive Maintenance Mini-Project

## 1. Topic Selection 
**Predictive maintenance of turbofan engines** using multivariate time-series sensor data to anticipate imminent failures and optimize maintenance scheduling.

---

## 2. Problem Definition 
The objective of this project is to **estimate the Remaining Useful Life (RUL)** of each engine from its sensor and operating-condition readings, then **flag engines as “failure imminent”** when their RUL drops below 20 cycles. By predicting failures ahead of time, maintenance can be scheduled proactively, reducing unplanned downtime and repair costs.

---

## 3. Background Study
Predictive maintenance is critical in industries like aerospace and manufacturing to lower operating costs and improve safety.  
- **NASA’s C-MAPSS benchmark** (Saxena et al., 2008) provides turbofan engine sensor readings up to failure.  
- **Machine-learning approaches** such as Random Forests have been applied successfully for RUL estimation (Babu et al., 2016).  
- **Deep-learning methods** (e.g. LSTMs) capture temporal dependencies in time-series data (Li et al., 2018).  
A brief review of these methods guided our choice of features and model.

---

## 4. Methodology or Approach 
- **Data preprocessing & labeling**  
  - Loaded raw readings from `PM_test.txt`, calculated each engine’s RUL, and created a binary “failure imminent” label (RUL < 20):contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}.  
- **Feature set**  
  - Three operating settings + 21 sensor measurements.  
- **Model pipeline**  
  - **Scaling**: `StandardScaler`  
  - **Classifier**: `RandomForestClassifier(n_estimators=100)`  
  - **Train/test split**: configurable test size & seed :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}  
- **Evaluation**  
  - Classification report (precision, recall, F1-score)  
  - Confusion matrix

---

## 5. Implementation or Analysis 
All code lives in `app.py`, powering a Streamlit dashboard that lets users:
1. **Upload & label data**  
2. **Explore data**:  
   - Dataframe previews & summary statistics  
   - RUL histograms and failure-class bar charts  
   - Sensor-correlation heatmap  
3. **Train & evaluate**:  
   - Interactive classification report and confusion-matrix plot  
4. **Inspect individual engines**: RUL-vs-cycle line charts  
5. **Export processed data** as CSV :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}

---

## 6. Results and Conclusion 
- **Model performance** (20-cycle threshold, 80/20 split, seed = 42):  
  - **Overall accuracy**: 87.3%  
  - **“Failure” class**:  
    - Precision = 79.2%  
    - Recall = 20.5%  
    - F1-score = 32.5%  
  - **Confusion matrix**:  
    ```
    Predicted ↓    No Failure    Failure
    Actual →                         
    No Failure      2208           21
    Failure          311           80
    ```
- **Challenges**  
  - Strong class imbalance → low recall for imminent failures  
  - Label granularity depends heavily on the chosen RUL threshold  
- **Future Work**  
  1. Address imbalance (e.g. SMOTE, class weights)  
  2. Hyperparameter tuning via cross-validation  
  3. Sequence models (LSTM/GRU) for temporal dynamics  
  4. Real-time deployment in a streaming pipeline

---
