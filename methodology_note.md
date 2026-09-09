# CPRI Data Challenge — Methodology Note
## Team Anveshan | BMS College of Engineering

---

## 1. Approach Used

Our solution is a **two-stage automated machine learning pipeline** designed to first isolate unreliable measurements and then predict the thermal Reference Parameter.

### Task 1 — Anomaly Detection (Valid / Invalid Classification)

We trained a **supervised XGBoost Gradient Boosting Classifier** on the 1,000 engineer-labelled historical records (866 Valid, 134 Invalid).

**Why supervised over unsupervised?** Unsupervised methods such as Isolation Forest flag any statistically unusual data point as an anomaly. In electrical testing, equipment genuinely operates in different thermal regimes under high-current or extended-duration conditions, producing legitimately extreme readings. A supervised classifier learns the *conditional* patterns that distinguish engineer-verified sensor failures from valid regime shifts—for example, it learns that high temperature rise is valid when accompanied by high load current, but invalid when load current is low (implying a sensor spike).

### Task 2 — Reference Parameter Prediction

We trained a **Gaussian Process Regressor (GPR)** exclusively on the **Valid** subset of historical data. GPR was selected for two critical reasons:

1. **Accuracy**: R² ≈ 0.998 on training data, with MAE approximately half that of gradient boosting alone.
2. **Built-in uncertainty quantification**: GPR natively outputs a calibrated standard deviation (σ) for each prediction, enabling principled identification of the "Three Test IDs requiring highest attention" without ad-hoc heuristics.

A **fallback XGBoost Regressor** was also trained as a guardrail against GPR's known weakness in extrapolation beyond the training distribution.

---

## 2. Parameters Considered Important

SHAP (SHapley Additive exPlanations) analysis identified the following hierarchy:

| Rank | Parameter | Rationale |
|------|-----------|-----------|
| 1 | **Load_Current_A** | Directly determines I²R resistive heating—the primary mechanism for thermal rise in electrical conductors. |
| 2 | **Sensor_S2** | Measures temperature rise near the outgoing/load-side terminal, physically closest to the hotspot. |
| 3 | **Sensor_S1** | Incoming terminal temperature—provides thermal gradient context. |
| 4 | **Sensor_S3** | Additional critical-location measurement capturing spatial heat distribution. |
| 5 | **Applied_Voltage_kV** | Determines dielectric and corona heating contributions at higher voltages. |
| 6 | **Test_Duration_min** | Controls thermal equilibrium approach—longer tests allow greater heat accumulation. |
| 7 | **Ambient_Temperature_C** | Baseline temperature offset; higher ambient reduces effective cooling. |

**Sensor_S4** was statistically assessed (Pearson r = −0.009 on Valid records) and **excluded** as irrelevant noise from the auxiliary condition-monitoring sensor.

---

## 3. Method for Detecting Abnormal Data

The XGBoost Classifier exploits **non-linear conditional interactions** between features:

- **Sensor spikes**: An extreme Sensor_S1 reading with low Load_Current is physically impossible (no heat source → no temperature rise). The decision trees learn these conditional bounds.
- **Missing critical data**: NaN values in S1/S2/S3, even after median imputation, leave residual statistical artefacts that split the tree toward "Invalid."
- **Physically inconsistent combinations**: Negative temperature rise values violate the Second Law of Thermodynamics for resistive heating under load and are flagged as recording errors.
- **Regime shifts preserved**: High temperatures under high current and long duration are classified as Valid, preserving genuine operating behaviour.

---

## 4. Assumptions Made

1. **Label trustworthiness**: The historical Valid/Invalid labels from CPRI engineers are assumed to correctly represent all relevant failure modes.
2. **Stationarity**: The physical relationship between electrical loading and thermal response of the test specimen has not permanently degraded between historical and new test campaigns.
3. **Sensor S4 irrelevance**: Based on statistical evidence (|r| < 0.01), S4 contributes no predictive signal and its inclusion would only add noise.
4. **Independent records**: Each test record represents an independent trial (no temporal autocorrelation between consecutive Test IDs).

---

## 5. Digital Twin Implementation

To automate this pipeline into a live Digital Twin for continuous operation, the following architecture is required:

### Step 1 — IoT Data Ingestion
Deploy edge sensors with MQTT/OPC-UA connectivity. Stream raw measurements to a cloud message broker (e.g., Google Cloud Pub/Sub or AWS IoT Core) for buffering and ordering.

### Step 2 — Real-Time Preprocessing
Use a stream processing engine (Apache Flink or Google Cloud Dataflow) to:
- Apply sliding-window median imputation for transient sensor dropouts
- Timestamp-align multi-sensor readings
- Compute derived features (e.g., I²t product)

### Step 3 — Model Microservices
Containerise the trained XGBoost Classifier and GPR Regressor as Docker microservices, deployed on Kubernetes or Cloud Run with auto-scaling. Expose via gRPC endpoints for low-latency inference (<50 ms per record).

### Step 4 — Two-Stage Inference Pipeline
Route each incoming test record through:
1. **Validity service** → flags sensor errors in real-time
2. **Regression service** → predicts Reference Parameter with uncertainty band

### Step 5 — Threshold Monitoring & Alerting
If the GPR predicted temperature exceeds the equipment's rated thermal limit, or if the prediction uncertainty (σ) exceeds a configured threshold, trigger automated alerts via webhooks/PagerDuty to the control room dashboard. Store all predictions and uncertainty bands for fleet-wide trend analysis and predictive maintenance scheduling.
