# Team8 - Image Retrieval System

Team8 project for C1 course of Master in Computer Vision (MCV)

## Quick Start

### 1. Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Image Retrieval System
```bash
python main.py
```

This will execute a **two-phase evaluation**:

#### **Phase 1: Validation (QSD1_W1)**
- Load BBDD (index) and QSD1_W1 (query) datasets with ground truth
- Run **Method 1**: CieLab Histogram + Histogram Intersection
- Run **Method 2**: HSV Histogram + Histogram Intersection  
- Evaluate both methods using mAP@1 and mAP@5 metrics
- Save validation results to `results/week1/QSD1_W1/` (with metrics.json)

#### **Phase 2: Test Predictions (QST1_W1)**
- Load BBDD (index) and QST1_W1 (query) datasets (no ground truth)
- Generate predictions for both methods
- Save test results to `results/week1/QST1_W1/` (only result.pkl files)
- No evaluation performed (no ground truth available)

## Results

The system generates results in the following structure:

### **Validation Results (QSD1_W1)**
```
results/week1/QSD1_W1/
├── method1/          # CieLab Histogram
│   ├── result.pkl    # Predictions (list of lists with BBDD image IDs)
│   └── metrics.json  # Performance metrics (mAP@1, mAP@5)
└── method2/          # HSV Histogram
    ├── result.pkl    # Predictions (list of lists with BBDD image IDs)
    └── metrics.json  # Performance metrics (mAP@1, mAP@5)
```

### **Test Results (QST1_W1)**
```
results/week1/QST1_W1/
├── method1/          # CieLab Histogram
│   └── result.pkl    # Test predictions (no ground truth for evaluation)
└── method2/          # HSV Histogram
    └── result.pkl    # Test predictions (no ground truth for evaluation)
```

### **Output Format**
- **result.pkl**: List of 30 queries, each containing top-10 BBDD image IDs
- **metrics.json**: Human-readable performance metrics (validation only)

## Dependencies

Core packages required:
- `numpy==1.24.4` - Numerical computing
- `opencv-python==4.8.1.78` - Colour space conversion
- `pillow==11.3.0` - Image loading

## Members

> Miguel Moral Hernández

> Miquel Sala Francí

> Roger Vendrell Colet
