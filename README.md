# parameter-efficient-fine-tuning-of-vlm
# Chest X-ray Analysis Tool

## Overview
The Chest X-ray Analysis Tool is an AI-powered web application built with Streamlit that analyzes chest X-ray images to detect 14 common thoracic pathologies. The tool uses a CLIP (Contrastive Language-Image Pre-training) vision-language model enhanced with LoRA (Low-Rank Adaptation) to interpret X-ray images and generate comprehensive radiological reports.



## Features
- **AI-Powered Image Analysis**: Uses a CLIP model with LoRA adaptation to detect 14 different chest pathologies
- **Ensemble Prediction**: Combines multiple image preprocessing techniques and text prompts for more reliable results
- **Detailed Medical Reports**: Generates structured reports with findings, impressions, and recommendations
- **Interactive Visualizations**: Displays prediction probabilities, disease categorizations, and analysis metrics
- **Confidence Metrics**: Provides uncertainty estimation and prediction quality assessment
- **Downloadable Results**: Export reports, visualizations, and findings data

## Pathologies Detected
The tool can detect the following chest conditions:
- Atelectasis (Collapsed/closed lung)
- Cardiomegaly (Enlarged heart)
- Effusion (Fluid around lung)
- Infiltration (Substance in lung tissue)
- Mass (Abnormal spot/mass)
- Nodule (Small round shadow)
- Pneumonia (Lung infection)
- Pneumothorax (Collapsed lung)
- Consolidation (Lung filled with fluid)
- Edema (Excess fluid in lung)
- Emphysema (Damaged air sacs)
- Fibrosis (Scarred lung tissue)
- Pleural_Thickening (Pleural thickening)
- Hernia (Protrusion through diaphragm)

## Technical Architecture
The application uses several key technologies:
- **CLIP Model**: OpenAI's vision-language model for image understanding
- **LoRA Adaptation**: Low-rank adaptation technique to enhance the CLIP model for medical imaging
- **Ensemble Methods**: Multiple preprocessing variations and prompt strategies for improved accuracy
- **Streamlit**: Web framework for the user interface
- **PyTorch**: Deep learning framework for model implementation
- **Transformers**: Hugging Face library for the CLIP model

### LoRA Implementation
The tool implements LoRA (Low-Rank Adaptation) for efficient fine-tuning of the CLIP model:
- Custom `LoRALinear` module for low-rank adaptation of linear layers
- Targeted application to specific modules in the CLIP architecture
- Configurable rank and scaling parameters

## Installation

### Prerequisites
- Python 3.7+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Dependencies
```bash
pip install -r requirements.txt
```

For optimal performance with GPU acceleration:
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu117
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/laasya2505/parameter-efficient-fine-tuning-of-vlm.git
cd chest-xray-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Launch the application** using the command above
2. **Upload a chest X-ray image** using the file uploader
3. **Click "Analyze Image"** to process the X-ray
4. **View the results** in the three tabs:
   - Medical Report: Detailed radiological report with findings and impressions
   - Visualizations: Interactive charts and visualizations of the predictions
   - Download: Export options for the report, visualizations, and data

### Adjusting Parameters
- Use the threshold slider in the sidebar to adjust the sensitivity of the detection

## How It Works

### Ensemble Prediction
The tool uses an ensemble approach that:
1. Applies multiple image preprocessing variations (contrast adjustment, equalization, etc.)
2. Uses different sets of text prompts (detailed medical, simple disease-focused, etc.)
3. Combines predictions with weighted averaging for more robust results
4. Estimates uncertainty using variance across ensemble members

### Report Generation
The medical report includes:
- Technique section
- Clinical information
- Findings organized by anatomical regions
- Impression with prioritized abnormalities
- Confidence assessment
- Limitations and disclaimer

### Visualization
The tool provides several visualizations:
- Probability bar chart for all pathologies
- Disease category heatmap
- Top findings summary

## Model Performance
- The model uses LoRA adaptation rather than full fine-tuning
- Performance depends on image quality and proper chest X-ray positioning
- The tool provides confidence metrics to indicate reliability of predictions

## Limitations and Disclaimer
- This tool is for **educational purposes only** and is not FDA-approved
- Not intended for clinical decision making or diagnosis
- Should not replace consultation with qualified healthcare providers
- Performance may vary significantly from expert radiological interpretation

## Future Enhancements
- Integration with DICOM format for medical images
- Support for lateral views and comparison with prior studies
- Fine-tuning on larger chest X-ray datasets
- Enhanced report customization
- Multi-language support



## Contributors
-Durga Sreelaasya Vemula
-Sanjana Sai Nallamalli
-Pradyumna Raju Birudaraju

## Acknowledgments
- CLIP model from OpenAI
- LoRA implementation inspired by Microsoft Research
- Streamlit for the web framework

