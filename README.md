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
- Tests basic image retrieval on QSD1_W1 → BBDD
- Uses LAB and HSV color histograms
- Applies histogram equalization preprocessing
- Saves results to `results/week_1/`

**Expected Output:**
```
🔎 IMAGE RETRIEVAL SYSTEM - WEEK 1
==================================================
✨ METHOD 1: LAB Histogram
✨ METHOD 2: HSV Histogram
✅ VALIDATION RESULTS
LAB:         mAP@1=0.XXX, mAP@5=0.XXX
HSV:         mAP@1=0.XXX, mAP@5=0.XXX
```

### Week 2 - Advanced Image Retrieval with Background Removal
```bash
python main_w2.py
```

**What it does:**
1. **Validation Phase**: Tests HSV and HSV Block Histograms on QSD1_W1 → BBDD
2. **Background Removal Phase**: Tests complete pipeline on QSD2_W2 → BBDD with background removal

**Expected Output:**
```
🔎 IMAGE RETRIEVAL SYSTEM - WEEK 2
==================================================
📊 VALIDATION PHASE (QSD1_W1)
✨ METHOD 1: HSV Histogram
✨ METHOD 2: HSV Block Histogram
✅ VALIDATION RESULTS
HSV:         mAP@1=0.XXX, mAP@5=0.XXX
HSV_BLOCKS:  mAP@1=0.XXX, mAP@5=0.XXX

🎭 BACKGROUND REMOVAL + IMAGE RETRIEVAL SYSTEM
==================================================
🔧 QSD2_W2 BACKGROUND REMOVAL + RETRIEVAL PIPELINE
🚀 Starting Background Removal + Image Retrieval Pipeline
📁 Loading datasets...
🎭 Applying background removal...
🔍 Computing descriptors...
🔎 Performing image retrieval...
📊 Evaluating retrieval performance...
🎯 Evaluating background removal quality...
💾 Saving results...
✅ Pipeline completed successfully!

📊 COMBINED RESULTS
🔍 RETRIEVAL PERFORMANCE:
  mAP@1: 0.XXX
  mAP@5: 0.XXX
🎭 BACKGROUND REMOVAL QUALITY:
  Precision: 0.XXX
  Recall:    0.XXX
  F1-Score:  0.XXX
```

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
│   └── QSD1_W1/
│       ├── method1/              # LAB histogram
│       │   ├── result.pkl
│       │   └── metrics.json
│       └── method2/              # HSV histogram
│           ├── result.pkl
│           └── metrics.json
└── week_2/
    ├── QSD1_W1/                  # Validation results
    │   ├── method_hsv/
    │   └── method_hsv_blocks/
    └── QSD2_W2/                  # Background removal results
        └── method_hsv_bg_removal/
            ├── result.pkl
            ├── retrieval_metrics.json
            └── background_removal_metrics.json
```

## 🎯 Key Features

### Week 1 Features
- Basic color histogram descriptors
- Histogram equalization preprocessing
- Standard image retrieval evaluation

### Week 2 Features
- **Spatial descriptors**: Block histograms and spatial pyramids
- **Background removal**: Automatic background removal for QSD2_W2
- **Dual evaluation**: Both retrieval performance and background removal quality
- **Flexible preprocessing**: Unified preprocessing system

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

## 📈 Performance Tips

1. **Use appropriate descriptors**: HSV works well for paintings, LAB for natural images
2. **Apply preprocessing**: Histogram equalization often improves performance
3. **Spatial descriptors**: Block histograms can capture spatial information
4. **Background removal**: Essential for QSD2_W2 dataset

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of a computer vision course. Please refer to the course guidelines for usage and distribution.