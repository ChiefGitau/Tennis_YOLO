# ML4QS Tennis Analysis Assignment

This directory contains Jupyter notebooks that demonstrate Machine Learning for Quantified Self (ML4QS) concepts applied to tennis video analysis.

## Notebooks Overview

### Chapter 2: Data Creation & Collection
**File:** `chapter2_data_creation_collection.ipynb`
- Video frame reading and temporal data extraction
- YOLO object detection for time-series position data
- Pre-processed detection data storage
- Temporal dataset creation with timestamps

### Chapter 3: Data Transformation & Preprocessing  
**File:** `chapter3_data_transformation_preprocessing.ipynb`
- Missing value imputation using pandas interpolation
- Rolling mean smoothing (equivalent to low-pass filtering)
- Butterworth low-pass filtering
- Coordinate transformations and normalization
- Outlier detection and removal

### Chapter 4: Feature Abstraction
**File:** `chapter4_feature_abstraction.ipynb`
- Temporal features: velocity and acceleration analysis
- Rolling window aggregation
- Pattern detection for ball hits using derivative analysis
- Frequency domain features using FFT
- Court coordinate transformations
- Multi-scale statistical features

### Chapter 5: Clustering & Pattern Recognition
**File:** `chapter5_clustering_pattern_recognition.ipynb`
- Ball hit detection using temporal pattern recognition
- K-means clustering of ball trajectory patterns
- DBSCAN clustering for event detection
- Hierarchical clustering for player positions
- Distance metrics implementation
- Temporal sequence clustering

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Data Requirements
The notebooks expect the tennis analysis data to be located at:
```
../tennis_analysis-main/
├── tracker_stubs/
│   ├── ball_detections.pkl
│   └── player_detections.pkl
├── input_videos/
│   └── input_video.mp4
└── models/
    └── yolo5_last.pt
```

### 3. Running the Notebooks
Start Jupyter:
```bash
jupyter notebook
```

Or use Jupyter Lab:
```bash
jupyter lab
```

## Key Concepts Demonstrated

### ML4QS Chapter 2 Concepts:
- Temporal data collection from video streams
- Creating structured datasets with timestamps
- Data quality assessment and validation

### ML4QS Chapter 3 Concepts:
- Missing value handling and interpolation
- Signal filtering and noise reduction
- Data normalization and transformation
- Outlier detection and correction

### ML4QS Chapter 4 Concepts:
- Temporal feature extraction
- Frequency domain analysis
- Pattern detection algorithms
- Multi-scale feature engineering

### ML4QS Chapter 5 Concepts:
- Unsupervised clustering techniques
- Distance metric implementations
- Pattern recognition in temporal data
- Event sequence analysis

## Expected Outputs

Each notebook generates:
- Visualizations of data processing steps
- Feature extraction results
- Clustering analysis results
- CSV files with processed data
- Performance metrics and validation results

## Notes

- The notebooks are designed to work with the tennis analysis project data
- Some cells may require significant computation time for video processing
- Pre-processed data (pickle files) are used to speed up development
- All visualizations are interactive and saved as outputs

## Troubleshooting

**Common Issues:**
1. **Import errors:** Ensure all requirements are installed
2. **File not found:** Check that tennis_analysis-main directory is at the correct relative path
3. **Memory issues:** Some operations may require significant RAM for video processing
4. **CUDA errors:** PyTorch/YOLO models may require GPU setup for optimal performance

**Performance Tips:**
- Use the pre-processed pickle files when available
- Limit video frame ranges for faster development
- Consider using smaller datasets for initial testing