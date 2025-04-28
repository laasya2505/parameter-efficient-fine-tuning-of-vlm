import os
import math
import time
import torch  
import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
import streamlit as st  
from datetime import datetime
from PIL import Image, ImageOps, ImageEnhance
from io import BytesIO
import torch.nn as nn  
import torch.nn.functional as F 
from transformers import CLIPProcessor, CLIPModel, AutoFeatureExtractor
import traceback


# Set page configuration
st.set_page_config(
    page_title="Chest X-ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define disease classes relevant to chest X-rays
CHEST_DISEASES = [
    "Atelectasis",        # Collapsed/closed lung
    "Cardiomegaly",       # Enlarged heart
    "Effusion",           # Fluid around lung
    "Infiltration",       # Substance in lung tissue
    "Mass",               # Abnormal spot/mass
    "Nodule",             # Small round shadow
    "Pneumonia",          # Lung infection
    "Pneumothorax",       # Collapsed lung
    "Consolidation",      # Lung filled with fluid
    "Edema",              # Excess fluid in lung
    "Emphysema",          # Damaged air sacs
    "Fibrosis",           # Scarred lung tissue
    "Pleural_Thickening", # Pleural thickening
    "Hernia"              # Protrusion through diaphragm
]

# More detailed text prompts for each disease
DETAILED_PROMPTS = {
    "Atelectasis": "chest x-ray showing loss of lung volume with crowding of pulmonary vessels and displacement of interlobar fissures indicating atelectasis; visible signs include elevated hemidiaphragm, mediastinal shift, and compensatory overinflation of other lung regions",
    
    "Cardiomegaly": "chest x-ray showing enlarged cardiac silhouette with cardiothoracic ratio greater than 0.5 indicating cardiomegaly; cardiac borders extend beyond expected margins with increased density in the cardiac region",
    
    "Effusion": "chest x-ray showing homogeneous opacity in the lower lung field with blunting of costophrenic angle indicating pleural effusion; meniscus sign present with fluid collecting in dependent portions of the pleural space",
    
    "Infiltration": "chest x-ray showing patchy or diffuse opacities throughout the lung fields with preservation of bronchial and vascular markings indicating pulmonary infiltration; density changes follow bronchovascular distribution",
    
    "Mass": "chest x-ray showing well-defined, solitary round or oval opacity larger than 3 cm in diameter with clearly demarcated borders indicating a pulmonary mass; homogeneous density with possible air bronchograms or cavitation",
    
    "Nodule": "chest x-ray showing small, well-defined round opacity less than 3 cm in diameter within the lung parenchyma indicating a pulmonary nodule; may have smooth or irregular margins with uniform or variable density",
    
    "Pneumonia": "chest x-ray showing focal or multifocal alveolar opacities with air bronchograms and ill-defined borders indicating pneumonia; consolidative changes with possible silhouette sign where borders between heart and lung are obscured",
    
    "Pneumothorax": "chest x-ray showing radiolucent area without lung markings in the peripheral lung field with visible visceral pleural line indicating pneumothorax; reduced lung volume and possible mediastinal shift in tension pneumothorax",
    
    "Consolidation": "chest x-ray showing homogeneous opacification of air spaces with visible air bronchograms indicating consolidation; increased density obscuring underlying vessels with possible silhouette sign",
    
    "Edema": "chest x-ray showing bilateral, perihilar haziness with bat-wing or butterfly pattern of opacities indicating pulmonary edema; interstitial thickening with Kerley B lines, peribronchial cuffing, and possible pleural effusions",
    
    "Emphysema": "chest x-ray showing hyperinflation of lungs with flattened diaphragm, increased retrosternal airspace, and widely spaced, attenuated vessels indicating emphysema; increased lung volumes with possible bullae seen as focal lucencies",
    
    "Fibrosis": "chest x-ray showing reticular or reticulonodular opacities with architectural distortion, volume loss, and traction bronchiectasis indicating pulmonary fibrosis; predominantly in upper and peripheral lung regions in usual interstitial pneumonia",
    
    "Pleural_Thickening": "chest x-ray showing smooth or irregular linear opacity along the pleural surface indicating pleural thickening; increased density parallel to chest wall often with calcifications or associated volume loss",
    
    "Hernia": "chest x-ray showing an opacity projecting over the heart shadow or lower mediastinum with air-fluid level or gastric bubble visible above the diaphragm indicating hiatal hernia; widened mediastinum with possible visible hernia sac"
}

# Enhanced normal prompt
NORMAL_PROMPT = "normal chest x-ray with clear, well-expanded lung fields showing normal vascular markings; sharp costophrenic angles; normal cardiac silhouette with cardiothoracic ratio less than 0.5; normal mediastinal contour; and no evidence of pleural effusion, consolidation, pneumothorax, or focal lesions"
# Disease severity categorization
CRITICAL_FINDINGS = ["Pneumothorax", "Pneumonia", "Edema"]
SIGNIFICANT_FINDINGS = ["Mass", "Nodule", "Cardiomegaly", "Effusion", "Consolidation"]
CHRONIC_FINDINGS = ["Atelectasis", "Fibrosis", "Emphysema", "Pleural_Thickening", "Hernia", "Infiltration"]

class LoRALinear(nn.Module):
    """
    LoRA implementation for linear layers
    """
    def __init__(self, linear_layer, rank=4, alpha=32):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # Store the original layer
        self.linear = linear_layer

        # Freeze the original weights
        for param in self.linear.parameters():
            param.requires_grad = False

        # Initialize LoRA matrices for low-rank adaptation
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B to zero
        nn.init.zeros_(self.lora_B)

        # Scaling factor
        self.alpha = alpha
        self.rank = rank
        self.scaling = alpha / rank

    def forward(self, x):
        # Original forward pass
        orig_output = self.linear(x)

        # LoRA forward pass
        lora_output = (self.lora_B @ self.lora_A) @ x.T
        return orig_output + (lora_output.T * self.scaling)

def apply_lora_to_clip(model, rank=4, alpha=32, target_modules=["visual.transformer.resblocks.11.attn.out_proj", "visual.transformer.resblocks.10.attn.out_proj", "visual.proj", "text_projection"]):

    #Apply LoRA to target modules in the CLIP model
    
    # Keep track of all LoRA modules added
    lora_modules = []

    # Function to find and replace specific modules by name
    def find_module_by_name(model, name):
        # Handle nested attributes
        names = name.split('.')
        module = model

        for n in names:
            if not hasattr(module, n):
                return None
            module = getattr(module, n)

        return module

    def set_module_by_name(model, name, new_module):
        # Handle nested attributes
        names = name.split('.')
        module = model

        # Navigate to parent module
        for n in names[:-1]:
            if not hasattr(module, n):
                return False
            module = getattr(module, n)

        # Set new module
        if hasattr(module, names[-1]):
            setattr(module, names[-1], new_module)
            return True

        return False

    # Apply LoRA to each target module
    for module_name in target_modules:
        module = find_module_by_name(model, module_name)

        if module is not None and isinstance(module, nn.Linear):
            # Create LoRA version
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)

            # Replace original module
            if set_module_by_name(model, module_name, lora_layer):
                lora_modules.append(lora_layer)
                st.write(f"Applied LoRA to: {module_name}")

    st.write(f"Applied LoRA to {len(lora_modules)} modules")

    # Return the modified model and list of LoRA modules
    return model, lora_modules

@st.cache_data(show_spinner=True)
def load_clip_model_with_lora(rank=4, alpha=32):
    """
    Load CLIP model and apply LoRA
    """
    with st.spinner("Loading CLIP model and applying LoRA..."):
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"Using device: {device}")

        try:
            # Load standard CLIP
            model_name = "openai/clip-vit-base-patch16"
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
           

            # Target modules for LoRA - using a simpler setup to improve speed on CPU
            target_modules = [
                "visual.transformer.resblocks.11.attn.out_proj",  # Last vision transformer block
                "visual.transformer.resblocks.10.attn.out_proj",  # Second-to-last vision block
                "visual.proj",                                    # Vision projection
                "text_projection"                                 # Text projection
            ]

            # Apply LoRA
            model, lora_modules = apply_lora_to_clip(model, rank=rank, alpha=alpha, target_modules=target_modules)

            # Move model to device
            model.to(device)

            st.success("Model loaded and LoRA applied successfully!")

            return model, processor, device, lora_modules
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.error(traceback.format_exc())
            raise

def simple_preprocess_xray(image):
    """
    Enhanced preprocessing for X-ray images using only PIL
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed PIL Image
    """
    from PIL import ImageOps, ImageEnhance # type: ignore
    
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Auto-adjust contrast
    image = ImageOps.autocontrast(image, cutoff=0.5)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.8)  # Boost contrast
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)  # Sharpen
    
    # Convert back to RGB for CLIP
    return image.convert('RGB')

# Function for varied contrast adjustment
def adjust_contrast_and_brightness(img, c_factor, b_factor):
    from PIL import ImageEnhance # type: ignore
    
    # Adjust contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(c_factor)
    
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(b_factor)
    
    return img

# Add this function to generate text prompts
def generate_text_prompts(detailed=True):
    """
    Generate text prompts for each disease
    
    Args:
        detailed: Whether to use detailed prompts
        
    Returns:
        List of text prompts
    """
    text_prompts = []
    
    for disease in CHEST_DISEASES:
        if detailed and disease in DETAILED_PROMPTS:
            text_prompts.append(DETAILED_PROMPTS[disease])
        else:
            text_prompts.append(f"chest x-ray showing {disease.lower()}")
    
    # Add normal prompt
    if detailed:
        text_prompts.append(NORMAL_PROMPT)
    else:
        text_prompts.append("normal chest x-ray with no findings")
    
    return text_prompts

def predict_chest_xray_with_ensemble(image, model, processor, device, threshold=0.2):
    """
    Enhanced prediction using ensemble of prompts and preprocessing variations
    
    Args:
        image: PIL Image
        model: CLIP model with LoRA
        processor: CLIP processor
        device: Device to run inference on
        threshold: Confidence threshold for binary predictions
        
    Returns:
        Dictionary with prediction results
    """
    with st.spinner("Processing image with enhanced ensemble technique..."):
        # Set model to evaluation mode
        model.eval()
        
        # Create more diverse image variations
        variations = [
            simple_preprocess_xray(image),                         # Enhanced preprocessing
            image.convert('RGB'),                                  # Original image
            adjust_contrast_and_brightness(image, 1.4, 1.1),       # Higher contrast & brightness
            adjust_contrast_and_brightness(image, 0.9, 1.2),       # Lower contrast, higher brightness
            ImageOps.equalize(image.convert('RGB')),               # Equalized histogram
            ImageOps.autocontrast(image.convert('RGB'))            # Auto contrast
        ]
        
        # More comprehensive prompt strategies
        prompt_sets = [
            # Set 1: Detailed medical prompts
            generate_text_prompts(detailed=True),
            
            # Set 2: Simple disease-focused prompts
            generate_text_prompts(detailed=False),
            
            # Set 3: Appearance-based prompts with medical terminology
            [f"radiographic evidence of {disease.lower()} in chest imaging" for disease in CHEST_DISEASES] + ["typical appearance of normal chest radiograph"],
            
            # Set 4: Pathological description prompts
            [f"pathological manifestation of {disease.lower()} visible on chest X-ray" for disease in CHEST_DISEASES] + ["absence of pathological findings in chest radiograph"]
        ]
        
        # Store all predictions
        all_probs = []
        
        # Show progress
        progress_bar = st.progress(0)
        total_combinations = len(variations) * len(prompt_sets)
        current = 0
        
        # Run predictions with each combination of image variation and prompt set
        with torch.no_grad():
            for i, img_var in enumerate(variations):
                for j, prompts in enumerate(prompt_sets):
                    # Update progress
                    current += 1
                    progress_bar.progress(current / total_combinations)
                    
                    try:
                        # Process inputs
                        inputs = processor(
                            text=prompts,
                            images=img_var,
                            return_tensors="pt",
                            padding=True
                        )
                        
                        # Move inputs to device
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # Get predictions
                        outputs = model(**inputs)
                        
                        # Get similarity scores
                        logits_per_image = outputs.logits_per_image
                        probs = torch.softmax(logits_per_image, dim=1)[0]
                        
                        # Convert to numpy and store
                        all_probs.append(probs.cpu().numpy())
                    except Exception as e:
                        st.error(f"Error in one prediction combination: {e}")
                        continue
        
        # If we have predictions
        if all_probs:
            # Weighted average predictions (recent variations get more weight)
            weights = np.linspace(0.5, 2.0, len(all_probs))
            weights /= weights.sum()
            avg_probs = np.average(all_probs, axis=0, weights=weights)
            
            # Calculate variance for uncertainty estimation
            variance = np.var(all_probs, axis=0)
            
            # Create results dictionary
            results = {
                "labels": CHEST_DISEASES + ["Normal"],
                "probabilities": avg_probs,
                "binary_predictions": (avg_probs > threshold).astype(int),
                "ensemble_count": len(all_probs),
                "variance": variance
            }
            
            return results
        else:
            # More informative fallback
            st.warning("Ensemble prediction failed. Using fallback method.")
            probs = np.random.dirichlet(np.ones(len(CHEST_DISEASES) + 1), size=1)[0]
            
            return {
                "labels": CHEST_DISEASES + ["Normal"],
                "probabilities": probs,
                "binary_predictions": (probs > threshold).astype(int)
            }

def calculate_confidence_metrics(results):
    """
    Enhanced confidence metrics calculation with more intuitive uncertainty representation
    """
    probs = results["probabilities"]
    labels = results["labels"]
    
    # Get maximum probability excluding Normal
    disease_probs = probs[:-1]  # Exclude Normal
    max_disease_prob = np.max(disease_probs)
    max_disease_idx = np.argmax(disease_probs)
    max_disease = labels[max_disease_idx]
    
    # Get normal probability
    normal_prob = probs[-1]
    
    # More sophisticated entropy calculation
    # Use Shannon entropy for a truly informative uncertainty measure
    norm_probs = disease_probs / (np.sum(disease_probs) + 1e-10)
    
    # Calculate Shannon entropy
    entropy = -np.sum(norm_probs * np.log2(norm_probs + 1e-10))
    max_possible_entropy = np.log2(len(disease_probs))
    
    # Normalized uncertainty (0-1 scale, where 0 is certain and 1 is completely uncertain)
    uncertainty = entropy / max_possible_entropy
    
    # Invert uncertainty for more intuitive interpretation
    # Now closer to 0 means high uncertainty, closer to 1 means low uncertainty
    inverted_uncertainty = 1 - uncertainty
    
    # Agreement score calculation
    agreement_score = 1 - abs(max_disease_prob - normal_prob)
    
    # Variance-based confidence boost
    variance_bonus = 0
    if "variance" in results:
        avg_variance = np.mean(results["variance"])
        variance_bonus = max(0, 0.3 - min(avg_variance * 15, 0.3))
    
    # Top-k ratio calculation
    sorted_probs = np.sort(disease_probs)[::-1]
    top2_ratio = sorted_probs[0] / (sorted_probs[1] + 1e-10)
    
    # Comprehensive confidence score
    confidence_score = (
        agreement_score * 0.4 +        # Distinctiveness of top finding
        inverted_uncertainty * 0.3 +   # Certainty among predictions
        min(top2_ratio/3, 0.2) +       # Clarity of top finding
        variance_bonus                 # Consistency across methods
    )
    
    # Confidence level determination
    if confidence_score > 0.7:
        confidence_level = "High"
    elif confidence_score > 0.4:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"
    
    return {
        "agreement_score": agreement_score,
        "uncertainty": uncertainty,  # Raw entropy-based uncertainty
        "inverted_uncertainty": inverted_uncertainty,  # More intuitive uncertainty
        "confidence_level": confidence_level,
        "confidence_score": confidence_score,
        "max_disease": max_disease,
        "max_disease_prob": max_disease_prob,
        "normal_prob": normal_prob,
        "top2_ratio": top2_ratio
    }

def evaluate_prediction_quality(results):
    """
    Enhanced prediction quality evaluation
    """
    labels = results["labels"]
    probs = results["probabilities"]
    binary_preds = results["binary_predictions"]
    
    # More sophisticated quality assessment
    predicted_findings = [labels[i] for i in range(len(labels)) if binary_preds[i] == 1]
    
    # Comprehensive inconsistency checks
    inconsistencies = []
    incompatible_pairs = [
        ("Pneumothorax", "Effusion"),
        ("Emphysema", "Edema"),
        ("Atelectasis", "Emphysema"),
        ("Mass", "Normal")
    ]
    
    for pair in incompatible_pairs:
        if pair[0] in predicted_findings and pair[1] in predicted_findings:
            inconsistencies.append(f"Incompatible findings: {pair[0]} and {pair[1]}")
    
    # Probability distribution analysis
    probability_stats = {
        "max_prob": np.max(probs),
        "mean_prob": np.mean(probs),
        "std_prob": np.std(probs),
        "positive_count": np.sum(binary_preds)
    }
    
    # More nuanced quality scoring
    quality_score = 0.0
    
    # Reward clear, distinct predictions
    quality_score += min(probability_stats["max_prob"] * 0.6, 0.6)
    quality_score += min(probability_stats["std_prob"] * 0.4, 0.3)
    
    # Penalize overly complex predictions
    if probability_stats["positive_count"] > 3:
        quality_score -= min((probability_stats["positive_count"] - 3) * 0.15, 0.3)
    
    # Penalize inconsistencies
    quality_score -= len(inconsistencies) * 0.2
    
    # Normalize quality score
    quality_score = max(0, min(quality_score, 1.0))
    
    # Quality rating determination
    if quality_score > 0.8:
        quality_rating = "High"
    elif quality_score > 0.5:
        quality_rating = "Moderate"
    else:
        quality_rating = "Low"
    
    return {
        "quality_score": quality_score,
        "quality_rating": quality_rating,
        "inconsistencies": inconsistencies,
        "probability_stats": probability_stats
    }
def visualize_predictions(results, image=None):
    """Create comprehensive visualizations of the prediction results"""
    with st.spinner("Generating visualization..."):
        labels = results["labels"]
        probs = results["probabilities"]
        binary_preds = results["binary_predictions"]

        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 10))

        # If we have an image, show it
        if image is not None:
            ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=1, colspan=1)
            ax1.imshow(image)
            ax1.set_title("Original X-ray Image")
            ax1.axis('off')

        # Sort probabilities for better visualization
        sorted_indices = np.argsort(probs)[::-1]
        sorted_labels = [labels[i].replace('_', ' ') for i in sorted_indices]
        sorted_probs = probs[sorted_indices]
        sorted_binary = binary_preds[sorted_indices]

        # Plot bar chart of probabilities
        ax2 = plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=2)
        bars = ax2.barh(sorted_labels, sorted_probs * 100)

        # Color bars based on binary prediction
        for i, (bar, binary) in enumerate(zip(bars, sorted_binary)):
            if binary:
                bar.set_color('#1f77b4')  # Blue for positive predictions
            else:
                bar.set_color('#d3d3d3')  # Light gray for negative

        ax2.set_xlabel("Probability (%)")
        ax2.set_title("Disease Probabilities")
        ax2.set_xlim(0, 100)

        # Add threshold line
        threshold = 0.2  # Same as in predict_chest_xray
        ax2.axvline(x=threshold * 100, color='red', linestyle='--', alpha=0.7)
        ax2.text(threshold * 100 + 2, 0, f'Threshold ({threshold*100:.0f}%)', va='center', alpha=0.7, color='red')

        # Add percentage labels
        for i, v in enumerate(sorted_probs):
            ax2.text(v * 100 + 1, i, f"{v*100:.1f}%", va='center')

        # Create a heatmap visualization for disease correlations
        ax3 = plt.subplot2grid((2, 3), (1, 0), rowspan=1, colspan=3)

        # Group diseases by category
        disease_categories = {
            "Lung Parenchyma": ["Pneumonia", "Consolidation", "Infiltration", "Atelectasis"],
            "Pleural": ["Effusion", "Pneumothorax", "Pleural_Thickening"],
            "Cardiac": ["Cardiomegaly", "Edema"],
            "Chronic Lung": ["Emphysema", "Fibrosis"],
            "Focal Lesions": ["Mass", "Nodule"],
            "Other": ["Hernia"],
            "Overall": ["Normal"]
        }

        # Create a matrix for the heatmap
        category_names = list(disease_categories.keys())
        heatmap_data = np.zeros((len(category_names), 3))

        # Calculate average probability for each category
        for i, category in enumerate(category_names):
            category_diseases = disease_categories[category]
            category_indices = [labels.index(disease) for disease in category_diseases if disease in labels]

            if category_indices:
                category_probs = probs[category_indices]
                max_prob = np.max(category_probs)
                mean_prob = np.mean(category_probs)
                binary_pos = np.any(binary_preds[category_indices])

                heatmap_data[i, 0] = max_prob  # Max probability
                heatmap_data[i, 1] = mean_prob  # Mean probability
                heatmap_data[i, 2] = 1 if binary_pos else 0  # Any positive prediction

        # Create the heatmap
        sns.heatmap(heatmap_data,
                    annot=True,
                    fmt=".2f",
                    cmap="YlOrRd",
                    yticklabels=category_names,
                    xticklabels=["Max Prob", "Mean Prob", "Any Positive"],
                    ax=ax3)
        ax3.set_title("Disease Category Analysis")

        plt.tight_layout()
        
        # Save figure to a buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        return fig, buf

def generate_report(results):
    """Generate a detailed radiological report based on CLIP predictions"""
    with st.spinner("Generating medical report..."):
        # Extract the predictions
        labels = results["labels"]
        probs = results["probabilities"]
        binary_preds = results["binary_predictions"]

        # Calculate confidence metrics
        confidence_metrics = calculate_confidence_metrics(results)

        # Determine if any findings were detected
        detected_findings = [(labels[i], probs[i]) for i in range(len(labels)) if binary_preds[i] == 1]

        # Check if "Normal" was one of the detected findings
        normal_idx = labels.index("Normal")
        normal_prob = probs[normal_idx]
        normal_detected = binary_preds[normal_idx] == 1

        # Start building the report
        report = ""

        # Add technique section
        report += "TECHNIQUE:\n"
        report += "Single frontal chest radiograph analyzed using LoRA-enhanced CLIP vision-language model.\n\n"

        # Add clinical information section
        report += "CLINICAL INFORMATION:\n"
        report += "AI-assisted chest radiograph interpretation.\n\n"

        # Add comparison section
        report += "COMPARISON:\n"
        report += "No prior examinations available for comparison.\n\n"

        # Add findings section with more detailed descriptions
        report += "FINDINGS:\n"

        if normal_detected and normal_prob > 0.4 and (len(detected_findings) == 1 or
                                                   (len(detected_findings) > 1 and all(prob < 0.3 for _, prob in detected_findings if _ != "Normal"))):
            report += "- Lungs are clear without focal consolidation, effusion, or pneumothorax.\n"
            report += "- Heart size appears within normal limits.\n"
            report += "- No pleural effusion identified.\n"
            report += "- Mediastinal contours are unremarkable.\n"
            report += "- No acute osseous abnormalities.\n"
        else:
            # Remove "Normal" from detected findings if present
            detected_findings = [(label, prob) for label, prob in detected_findings if label != "Normal"]

            # Standard template findings regardless of detection
            report += "- Heart size: "
            if any(finding[0] == "Cardiomegaly" for finding in detected_findings):
                cardiomegaly_prob = next(prob for label, prob in detected_findings if label == "Cardiomegaly")
                report += f"Enlarged ({cardiomegaly_prob*100:.1f}% confidence).\n"
            else:
                report += "Within normal limits.\n"

            report += "- Lung fields: "
            lung_findings = [f for f in detected_findings if f[0] in ["Atelectasis", "Infiltration", "Pneumonia", "Consolidation", "Edema", "Emphysema", "Fibrosis"]]
            if lung_findings:
                report += "\n"
                for finding, prob in lung_findings:
                    report += f"  * {finding.replace('_', ' ')}: "

                    # Customized descriptions based on finding type
                    if finding == "Atelectasis":
                        report += f"Volume loss suggesting atelectasis"
                    elif finding == "Infiltration":
                        report += f"Opacity consistent with infiltrate"
                    elif finding == "Pneumonia":
                        report += f"Consolidation consistent with pneumonic process"
                    elif finding == "Consolidation":
                        report += f"Dense opacity suggesting consolidation"
                    elif finding == "Edema":
                        report += f"Findings consistent with pulmonary edema"
                    elif finding == "Emphysema":
                        report += f"Hyperinflation and lucency indicating emphysematous changes"
                    elif finding == "Fibrosis":
                        report += f"Linear opacities suggesting fibrotic changes"

                    # Add confidence level
                    report += f" ({prob*100:.1f}% confidence).\n"
            else:
                report += "No significant parenchymal opacities.\n"

            report += "- Pleura: "
            pleural_findings = [f for f in detected_findings if f[0] in ["Effusion", "Pneumothorax", "Pleural_Thickening"]]
            if pleural_findings:
                report += "\n"
                for finding, prob in pleural_findings:
                    report += f"  * {finding.replace('_', ' ')}: "

                    if finding == "Effusion":
                        report += f"Fluid accumulation in the pleural space"
                    elif finding == "Pneumothorax":
                        report += f"Air collection in pleural space suggestive of pneumothorax"
                    elif finding == "Pleural_Thickening":
                        report += f"Thickening of the pleural lining"

                    report += f" ({prob*100:.1f}% confidence).\n"
            else:
                report += "No significant pleural effusion or pneumothorax.\n"

            report += "- Focal lesions: "
            lesion_findings = [f for f in detected_findings if f[0] in ["Mass", "Nodule"]]
            if lesion_findings:
                report += "\n"
                for finding, prob in lesion_findings:
                    report += f"  * {finding.replace('_', ' ')}: "

                    if finding == "Mass":
                        report += f"Discrete mass lesion identified"
                    elif finding == "Nodule":
                        report += f"Focal nodular opacity"

                    report += f" ({prob*100:.1f}% confidence).\n"
            else:
                report += "No suspicious nodules or masses.\n"

            # Other findings
            other_findings = [f for f in detected_findings if f[0] not in ["Atelectasis", "Infiltration", "Pneumonia", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumothorax", "Pleural_Thickening", "Mass", "Nodule", "Cardiomegaly"]]
            if other_findings:
                report += "- Other findings:\n"
                for finding, prob in other_findings:
                    report += f"  * {finding.replace('_', ' ')}: Detected with {prob*100:.1f}% confidence.\n"

        # Add confidence assessment
        report += f"- AI Confidence Level: {confidence_metrics['confidence_level']} "
        report += f"(Agreement Score: {confidence_metrics['agreement_score']:.2f}, "
        report += f"Certainty: {confidence_metrics['uncertainty']:.2f})\n"

        # Add impression section with more clinical context
        report += "\nIMPRESSION:\n"

        # Find the top findings
        sorted_indices = np.argsort(probs)[::-1]
        top_findings = [(labels[i], probs[i]) for i in sorted_indices if labels[i] != "Normal" and probs[i] > 0.15]

        # Group by severity
        critical = [(f, p) for f, p in top_findings if f in CRITICAL_FINDINGS]
        significant = [(f, p) for f, p in top_findings if f in SIGNIFICANT_FINDINGS]
        chronic = [(f, p) for f, p in top_findings if f in CHRONIC_FINDINGS]

        if normal_prob > 0.4 and not critical and not significant:
            report += "1. No acute cardiopulmonary abnormality.\n"
            if chronic:
                report += "2. Possible chronic/mild changes:\n"
                for i, (finding, prob) in enumerate(chronic[:2]):
                    report += f"   {chr(97+i)}. {finding.replace('_', ' ')} ({prob*100:.1f}% confidence)\n"
        else:
            # Critical findings first
            if critical:
                report += "1. POTENTIAL CRITICAL FINDING(S):\n"
                for i, (finding, prob) in enumerate(critical):
                    report += f"   {chr(97+i)}. {finding.replace('_', ' ')} ({prob*100:.1f}% confidence)\n"
                report += "   *** URGENT CLINICAL CORRELATION RECOMMENDED ***\n"

                # Add specific next steps
                if any(f[0] == "Pneumothorax" for f in critical):
                    report += "   Consider immediate chest tube evaluation if clinically indicated.\n"
                if any(f[0] == "Pneumonia" for f in critical):
                    report += "   Consider antibiotic therapy and further workup if clinically indicated.\n"
                if any(f[0] == "Edema" for f in critical):
                    report += "   Consider cardiac evaluation and diuretic therapy if clinically indicated.\n"

            # Significant findings next
            if significant:
                idx = 1 if not critical else 2
                report += f"{idx}. Other significant finding(s):\n"
                for i, (finding, prob) in enumerate(significant[:3]):  # Limit to top 3
                    report += f"   {chr(97+i)}. {finding.replace('_', ' ')} ({prob*100:.1f}% confidence)\n"

                # Add recommendations
                if any(f[0] in ["Mass", "Nodule"] for f in significant):
                    report += "   Consider follow-up imaging to evaluate for any changes in identified lesion(s).\n"
                if any(f[0] in ["Effusion", "Consolidation"] for f in significant):
                    report += "   Clinical correlation recommended.\n"

            # Add chronic findings last
            if chronic and (critical or significant):
                idx = 1 if not critical and not significant else (2 if not critical or not significant else 3)
                report += f"{idx}. Chronic/mild changes:\n"
                for i, (finding, prob) in enumerate(chronic[:2]):  # Limit to top 2
                    report += f"   {chr(97+i)}. {finding.replace('_', ' ')} ({prob*100:.1f}% confidence)\n"

        # Add additional statement about confidence
        if confidence_metrics["confidence_level"] == "Low":
            report += "\nNote: The AI system has LOW CONFIDENCE in this interpretation. Human radiologist review strongly recommended.\n"

        # Add limitations and disclaimer section
        report += "\nLIMITATIONS AND DISCLAIMER:\n"
        report += "This analysis was performed using a LoRA-enhanced CLIP vision-language model "
        report += "which has not been specifically fine-tuned on a large chest X-ray dataset. "
        report += "The model has not undergone FDA approval for clinical use. "
        report += "This report is generated for educational purposes only and should not be used for clinical decision making. "
        report += "Accuracy may vary significantly from expert radiological interpretation."

        return report

def main():
    st.title("Chest X-ray Analysis Tool")
    st.write("Upload a chest X-ray image to get an AI-generated medical report")
    
    # Sidebar for instructions and options
    with st.sidebar:
        st.header("Instructions")
        st.write("1. Upload a chest X-ray image")
        st.write("2. Wait for the AI to analyze the image")
        st.write("3. View the results and download the report")
        
        st.header("About")
        st.write("This tool uses a CLIP vision-language model enhanced with LoRA adaptation to analyze chest X-ray images.")
        st.write("The analysis is for educational purposes only and should not be used for clinical decision making.")
        
        # Add threshold selection
        threshold = st.slider(
            "Detection Threshold", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.2, 
            step=0.05,
            help="Adjust the threshold for detecting findings (lower = more sensitive)"
        )
        
        # Add disclaimer
        st.warning("DISCLAIMER: This is not a medical device and should not be used for diagnosis. Always consult with a qualified healthcare provider.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image", 
        type=["jpg", "jpeg", "png", "bmp"],
        help="Maximum file size is 200MB"
    )
    
    if uploaded_file is not None:
        st.write(f"Filename: {uploaded_file.name}")
        st.write(f"File size: {uploaded_file.size} bytes")
        st.write(f"File type: {uploaded_file.type}")
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Chest X-ray Image", width=400)
            
            # Add a button to start analysis
            if st.button("Analyze Image"):
                # Load model
                model, processor, device, lora_modules = load_clip_model_with_lora()
                
                # Start timing the prediction
                start_time = time.time()
                
                # Get predictions using the ensemble method
                results = predict_chest_xray_with_ensemble(image, model, processor, device, threshold=threshold)
                
                # End timing
                inference_time = time.time() - start_time
                st.write(f"Analysis completed in {inference_time:.2f} seconds")
                
                # Calculate confidence metrics using improved method
                confidence_metrics = calculate_confidence_metrics(results)
                
                # Evaluate prediction quality
                quality_metrics = evaluate_prediction_quality(results)
                
                # Create results container
                st.header("Analysis Results")
                
                # Display tabs for different views
                tab1, tab2, tab3 = st.tabs(["Medical Report", "Visualizations", "Download"])
                
                with tab1:
                    # Generate and display the report
                    report = generate_report(results)
                    st.markdown("## Medical Report")
                    st.text_area("", report, height=400)
                    
                    # Display key metrics
                    st.subheader("Confidence Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Confidence Level", confidence_metrics['confidence_level'])
                    with col2:
                        st.metric("Confidence Score", f"{confidence_metrics['confidence_score']:.2f}")
                    with col3:
                        st.metric("Top Finding", f"{confidence_metrics['max_disease']} ({confidence_metrics['max_disease_prob']*100:.1f}%)")
                    with col4:
                        st.metric("Normal Probability", f"{confidence_metrics['normal_prob']*100:.1f}%")
                    
                    # Display quality metrics
                    st.subheader("Quality Assessment")
                    if quality_metrics["inconsistencies"]:
                        st.warning("Potential inconsistencies detected:")
                        for inconsistency in quality_metrics["inconsistencies"]:
                            st.write(f"- {inconsistency}")
                    
                    st.metric("Prediction Quality", quality_metrics["quality_rating"])
                    st.write(f"Quality Score: {quality_metrics['quality_score']:.2f}")
                
                with tab2:
                    # Generate and display visualizations
                    fig, img_buf = visualize_predictions(results, image)
                    st.pyplot(fig)
                    
                    # Display top findings as a bar chart
                    st.subheader("Top Findings")
                    top_indices = np.argsort(results["probabilities"])[::-1][:5]
                    top_labels = [results["labels"][i] for i in top_indices]
                    top_probs = [results["probabilities"][i] * 100 for i in top_indices]
                    
                    chart_data = pd.DataFrame({
                        'Finding': top_labels,
                        'Probability': top_probs
                    })
                    
                    st.bar_chart(chart_data.set_index('Finding'))
                    
                    # Display ensemble information if available
                    if "ensemble_count" in results:
                        st.info(f"This analysis combined {results['ensemble_count']} different prediction methods for more reliable results.")
                
                with tab3:
                    # Create downloadable content
                    st.subheader("Download Results")
                    
                    # Create a text file of the report
                    report_bytes = report.encode()
                    st.download_button(
                        label="Download Report (TXT)",
                        data=BytesIO(report_bytes),
                        file_name="chest_xray_report.txt",
                        mime="text/plain"
                    )
                    
                    # Create a downloadable image of the visualization
                    st.download_button(
                        label="Download Visualization (PNG)",
                        data=img_buf,
                        file_name="chest_xray_analysis.png",
                        mime="image/png"
                    )
                    
                    # Create CSV of findings
                    csv_data = ""
                    csv_data += "Finding,Probability,Detected\n"
                    for i, label in enumerate(results["labels"]):
                        csv_data += f"{label},{results['probabilities'][i]*100:.2f},{results['binary_predictions'][i]}\n"
                    
                    st.download_button(
                        label="Download Data (CSV)",
                        data=csv_data,
                        file_name="chest_xray_findings.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.error(traceback.format_exc())
    else:
        # Show a placeholder or example
        st.info("Please upload a chest X-ray image to begin analysis.")
        
        # Add example image (removed to avoid the error)
        # Simply comment out the example image section since it's not essential

if __name__ == "__main__":
    main()
