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

This will:
- Load BBDD and QSD1 datasets
- Compute RGB and HSV histogram descriptors
- Evaluate both methods
- Save results in a .pkl to `results/week1/QST1/` directory
- Display performance metrics (mAP@1, mAP@5)

## Results

The system generates results in the following structure:
```
results/week1/QST1/
├── method1/          # RGB Histogram
│   ├── result.pkl    # Competition results (binary)
│   └── metrics.json  # Human-readable metrics
└── method2/          # HSV Histogram (best performing)
    ├── result.pkl    # Competition results (binary)
    └── metrics.json  # Human-readable metrics
```

## Dependencies

Core packages required:
- `numpy==1.24.4` - Numerical computing
- `opencv-python==4.8.1.78` - HSV color space conversion
- `pillow==11.3.0` - Image processing

## Testing

This project includes integration tests:


### Integration Scripts

Comprehensive tests that use real datasets and test the complete system. Run individually:

```bash
# Test BBDD dataset loading
python tests/integration/test_load_BBDD.py

# Test QSD1 dataset loading  
python tests/integration/test_load_qsd1_w1.py

# Test histogram descriptors
python tests/integration/test_histograms.py
```

Integration scripts are located in `tests/integration/` and test:
- Dataset loading and validation
- Histogram computation with real images


