## Chronic Kidney Disease (CKD) Prediction API

This API empowers healthcare professionals and researchers with the ability to predict Chronic Kidney Disease (CKD) stages and classes by leveraging machine learning models. It analyzes provided patient data to estimate the class (severity) and stage (progression) of CKD.

**All requests and responses are formatted in JSON.**

## Endpoints

### POST /detect_cdk & POST /cdk_stage

These endpoints work sequentially to predict CKD class and stage. 

```json
// POST /detect_cdk Request Body
{
  "blood_pressure": float (mmHg),  // Systolic and diastolic blood pressure combined (consider combining for implementation)
  "specific_gravity": float,      // Urine specific gravity
  "albumin": float (mg/dL),         // Serum albumin level
  "sugar": float (mg/dL) (optional),           // Blood sugar level
  "red_blood_cell": float (T/L),    // Red blood cell count
  "blood_urea": float (mg/dL),       // Blood urea nitrogen level
  "serum_creatinine": float (mg/dL), // Serum creatinine level
  "sodium": float (mEq/L),          // Serum sodium level
  "potassium": float (mEq/L),        // Serum potassium level
  "hemoglobin": float (g/dL),        // Hemoglobin level
  "white_blood_cell_count": float (T/L), // White blood cell count
  "red_blood_cell_count": float (T/L) (redundant, can be omitted)
}
```

```json
// POST /detect_cdk Response
{
  "class": int (0-5),  // Predicted CKD class (0: No CKD, 1-5: Increasing severity)
  "status": True      // API call successful
}
```

```json
// POST /get_stage Request Body
{
  "blood_pressure": float (mmHg),
  "specific_gravity": float,
  "albumin": float (mg/dL),
  "sugar": float (mg/dL) (optional),
  "red_blood_cell": float (T/L),
  "blood_urea": float (mg/dL),
  "serum_creatinine": float (mg/dL),
  "sodium": float (mEq/L),
  "potassium": float (mEq/L),
  "hemoglobin": float (g/dL),
  "white_blood_cell_count": float (T/L),
  "red_blood_cell_count": float (T/L) (redundant, can be omitted),
  "cdk_class": int (0-5)  // Required: Predicted CKD class from /detect_cdk
}
```

```json
// POST /get_stage Response
{
  "stage": int (1-5),  // Predicted CKD stage (1: Early, 5: Kidney failure)
  "status": True      // API call successful
}
```