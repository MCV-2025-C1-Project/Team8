# Computer Vision Project - Image Retrieval System

This project implements a comprehensive image retrieval system with background removal capabilities for computer vision applications.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Team8

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Running the Experiments

### Week 1 - Basic Image Retrieval
```bash
python main_w1.py
```

**What it does:**
- Tests LAB and HSV color histograms on QSD1_W1 → BBDD (validation)
- Tests LAB and HSV color histograms on QST1_W1 → BBDD (test)
- Uses gamma correction for LAB, histogram equalization for HSV
- Saves results to `results/week_1/`

### Week 2 - Advanced Image Retrieval with Background Removal
```bash
python main_w2.py
```

**What it does:**
1. **Validation Phase 1**: Tests HSV and HSV Block Histograms on QSD1_W1 → BBDD
2. **Validation Phase 2**: Tests K-Means background removal + HSV Block Histograms on QSD2_W2 → BBDD
3. **Test Phase**: Tests HSV Block Histograms on QST1_W2 and QST2_W2 (with background removal)

## 📁 Project Structure

```
Team8/
├── data/                           # Dataset files
│   ├── qsd1_w1/                   # Week 1 query dataset
│   ├── qsd2_w2/                   # Week 2 query dataset (with background removal)
│   └── BBDD/                      # Index dataset
├── dataloader/                    # Dataset loading utilities
│   └── dataloader.py
├── descriptors/                   # Image descriptor methods
│   ├── descriptors.py            # Main descriptor enum
│   ├── color_histograms.py       # Color histogram functions
│   └── spatial_histograms.py    # Spatial histogram functions
├── preprocessing/                 # Image preprocessing
│   ├── preprocessors.py         # Unified preprocessing enum
│   ├── color_adjustments.py     # Color adjustment functions
│   └── background_removers.py   # Background removal methods
├── services/                      # Main system services
│   ├── image_retrieval_system.py           # Basic retrieval system
│   └── background_removal_image_retrieval_system.py  # Combined system
├── utils/                         # Utility functions
│   ├── metrics.py               # Evaluation metrics
│   ├── measures.py              # Similarity measures
│   └── spatial.py               # Spatial processing
├── main_w1.py                    # Week 1 experiments
├── main_w2.py                    # Week 2 experiments
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🔧 System Components

### Descriptor Methods
- **Color Histograms**: Grayscale, RGB, LAB, HSV
- **Block Histograms**: Spatial partitioning with color histograms
- **Spatial Pyramid**: Multi-resolution spatial histograms

### Preprocessing Methods
- **Color Adjustments**: Histogram equalization, gamma correction, Gaussian blur, median filter
- **Background Removal**: K-means clustering, rectangle-based removal

### Similarity Measures
- **Histogram Intersection**: For histogram-based descriptors
- **Euclidean Distance**: For feature vectors
- **Cosine Similarity**: For normalized features

## 📊 Results Structure

Results are saved in the following structure:
```
results/
├── week_1/
│   ├── QSD1_W1/                  # Validation results
│   │   ├── method_lab/
│   │   └── method_hsv/
│   └── QST1_W1/                  # Test results
│       ├── method_lab/
│       └── method_hsv/
└── week_2/
    ├── QSD1_W1/                  # Validation results
    │   ├── method_hsv/
    │   └── method_hsv_blocks/
    ├── QSD2_W2/                  # Background removal results
    │   └── method_hsv_blocks_kmeans/
    ├── QST1_W2/                  # Test results
    │   └── method_hsv_blocks/
    └── QST2_W2/                  # Test results with background removal
        └── method_hsv_blocks_kmeans/
            ├── result.pkl
            └── 00000.png, 00001.png, ...  # Predicted masks
```

## 🎯 Key Features

### Week 1 Features
- LAB and HSV color histogram descriptors
- Gamma correction and histogram equalization preprocessing
- Validation and test phase evaluation

### Week 2 Features
- **Spatial descriptors**: HSV Block Histograms with configurable block sizes
- **Background removal**: K-Means clustering for QSD2_W2 and QST2_W2
- **Dual evaluation**: Both retrieval performance and background removal quality
- **Multiple datasets**: QSD1_W1, QSD2_W2, QST1_W2, QST2_W2

## 🔍 Understanding the Results

### Retrieval Metrics
- **mAP@1**: Mean Average Precision at rank 1 (top result accuracy)
- **mAP@5**: Mean Average Precision at rank 5 (top 5 results accuracy)

### Background Removal Metrics
- **Precision**: Percentage of predicted foreground pixels that are correct
- **Recall**: Percentage of actual foreground pixels that are detected
- **F1-Score**: Harmonic mean of precision and recall

## 🛠️ Customization

### Adding New Descriptors
1. Add new method to `DescriptorMethod` enum in `descriptors/descriptors.py`
2. Implement the computation logic
3. Update the `compute()` method

### Adding New Preprocessing
1. Add new method to `PreprocessingMethod` enum in `preprocessing/preprocessors.py`
2. Implement the `apply()` method
3. Add any required parameters

### Modifying Experiments
Edit `main_w1.py` or `main_w2.py` to:
- Change descriptor methods
- Modify preprocessing parameters
- Add new evaluation metrics
- Test different datasets

## 🐛 Troubleshooting

### Common Issues
1. **Import errors**: Make sure all dependencies are installed (`pip install -r requirements.txt`)
2. **Dataset not found**: Ensure data files are in the correct `data/` directory
3. **Memory issues**: Reduce batch sizes or use smaller datasets
4. **Performance issues**: Try different descriptor methods or preprocessing

### Debug Mode
Add debug prints to understand the pipeline:
```python
# In main_w2.py, add debug prints
print(f"Query dataset size: {len(query_dataset.data)}")
print(f"Index dataset size: {len(index_dataset.data)}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of a computer vision course. Please refer to the course guidelines for usage and distribution.