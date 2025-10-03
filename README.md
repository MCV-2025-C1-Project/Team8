# Team8

Team8 project for C1 course of Master in Computer Vision (MCV)

## Dependencies

All the needed dependencies are listed in the _requirements.txt_ file.

Ideally, install them within a virtual environment. Create and activate it with:

```bash
python3 -m venv venv/
source myenv/bin/activate
# Windows: venv\Scripts\Activate
```

And then, from the root of the project, install the dependencies with:

```bash
pip install -r requirements.txt
```

Update the requirements.txt running:

```bash
pip freeze > requirements.txt
```

## Testing

This project includes two types of tests:

### Unit Tests

Fast, isolated tests for individual functions and classes. Run with pytest:

```bash
# Run all unit tests
python -m pytest tests/unit/

# Run with verbose output
python -m pytest tests/unit/ -v

# Run specific test file
python -m pytest tests/unit/test_measures.py
```

Unit tests are located in `tests/unit/` and test:
- Distance measures (euclidean, L1, chi-squared, histogram intersection, hellinger kernel)
- Metrics functions (APK, mAP@K)

### Integration Scripts

Comprehensive tests that use real datasets and test the complete system. Run individually:

```bash
# Test BBDD dataset loading
python tests/integration/test_load_BBDD.py

# Test QSD1 dataset loading  
python tests/integration/test_load_qsd1_w1.py

# Test histogram descriptors
python tests/integration/test_histograms.py

# Test distance measures with real data
python tests/integration/test_measures.py
```

Integration scripts are located in `tests/integration/` and test:
- Dataset loading and validation
- Histogram computation with real images
- Distance measure calculations with actual data
- Complete image retrieval pipeline

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```
