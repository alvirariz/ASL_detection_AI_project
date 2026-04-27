import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# ===== MODEL ARCHITECTURE =====
class DeepCNN(nn.Module):
    def __init__(self, dropout_rate, dense_units):
        super(DeepCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU()
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, 29)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x

# ===== ASL LETTER MAPPING =====
ASL_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
               'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# ===== CUSTOM CSS =====
# Replace the CSS section with this updated version:

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #7A8F72;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] strong,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF !important;
    }
            

   

    /* Uploaded file chip - light styling */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploader"] div[class*="uploadedFile"],
    [data-testid="stFileUploader"] li,
    [data-testid="stFileUploader"] [class*="st-emotion"] {
        background-color: #E8F0E5 !important;
        color: #2C3E2A !important;
        border-radius: 8px !important;
    }

    /* File name and size text inside chip */
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] div {
        color: #2C3E2A !important;
        background-color: transparent !important;
    }

    /* The X/close button on the chip */
    [data-testid="stFileUploader"] button[kind="secondary"],
    [data-testid="stFileUploader"] button {
        background-color: #7A8F72 !important;
        color: white !important;
        border: none !important;
    }
    
    /* Main content text - DARK for readability */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #2C3E2A !important;
        font-weight: 600;
    }
    
    .stApp p, .stApp span, .stApp div {
        color: #2C3E2A !important;
    }
    
    /* File uploader - LIGHTER */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border-radius: 8px;
        padding: 1.5rem;
        border: 2px dashed #7A8F72;
    }
    
    [data-testid="stFileUploader"] section {
        background-color: #F8FAF7 !important;
        border-radius: 8px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"] section button {
        background-color: #7A8F72 !important;
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    
    [data-testid="stFileUploader"] section button:hover {
        background-color: #5F7359 !important;
    }
    
    [data-testid="stFileUploader"] small {
        color: #5F7359 !important;
    }
    
    /* Cards/Containers */
    .prediction-card {
        background-color: #F8FAF7;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #D4E0D0;
    }
    
    .info-card {
        background-color: #E8F0E5;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #7A8F72;
        color: #2C3E2A !important;
    }
    
    .info-card p, .info-card strong {
        color: #2C3E2A !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #7A8F72;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background-color: #5F7359;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #7A8F72;
    }
    
    /* Prediction display */
    .prediction-letter {
        font-size: 5rem;
        font-weight: 700;
        color: #2C3E2A;
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #E8F0E5 0%, #D4E0D0 100%);
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid #7A8F72;
    }
    
    .confidence-score {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2C3E2A;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Top predictions */
    .prediction-item {
        background-color: #F8FAF7;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        border-left: 4px solid #7A8F72;
        border: 1px solid #D4E0D0;
    }
    
    .prediction-item span {
        color: #2C3E2A !important;
        font-size: 1.1rem;
    }
    
    /* Caption text */
    .stCaption {
        color: #5F7359 !important;
    }
</style>
""", unsafe_allow_html=True)

# ===== LOAD MODEL =====
@st.cache_resource
def load_model():
    import os
    model_path = os.path.join(os.path.dirname(__file__), 'model_v3.pth')
    model = DeepCNN(dropout_rate=0.5, dense_units=256)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# ===== PREPROCESS IMAGE =====
def preprocess_image(image):
    """
    Preprocess uploaded image to match training data format
    """
    img = image.convert('L')
    img = img.resize((64, 64))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img_tensor

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="ASL Sign Language Recognition",
    page_icon="✋",
    layout="wide"
)

# ===== SIDEBAR =====
with st.sidebar:
    st.title("Model Information")
    st.markdown("---")
    
    st.markdown("""
    **Architecture**  
    Deep Convolutional Neural Network
    
    **Performance**  
    Validation Accuracy: 99.90%
    
    **Classes**  
    29 ASL Letters (A-Z + Special)
    
    **Hyperparameters**  
    • Dropout Rate: 0.5  
    • Dense Units: 256  
    • Optimizer: Adam  
    • Learning Rate: 0.001
    """)
    
    st.markdown("---")
    
    st.markdown("### How to Use")
    st.markdown("""
    1. Upload hand gesture image
    2. Wait for processing
    3. View predicted letter
    4. Check confidence score
    5. Review alternatives
    """)
    
    st.markdown("---")
    
    st.markdown("### About ASL")
    st.markdown("""
    American Sign Language uses hand shapes and movements to communicate. This model recognizes static hand gestures for the alphabet.
    """)

# ===== MAIN CONTENT =====
st.markdown("<h1 style='text-align: center; color: #2C3E2A;'>ASL Sign Language Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5F7359; font-size: 1.2rem;'>Upload an image of an ASL hand gesture for instant recognition</p>", unsafe_allow_html=True)
st.markdown("---")

# Two column layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<h3 style='color: #FFFFFF;'>Upload Image</h3>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        st.caption(f"Image size: {image.size[0]} × {image.size[1]} pixels")
    else:
        st.markdown("""
        <div class='info-card'>
            <p style='margin: 0; text-align: center; color: #FFFFF;'><strong>No image uploaded</strong></p>
            <p style='margin: 0.5rem 0 0 0; text-align: center; color: #2C3E2A;'>Click above to select an ASL gesture image</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("<h3 style='color: #2C3E2A;'>Prediction Results</h3>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        with st.spinner('Loading model...'):
            model = load_model()
        
        with st.spinner('Analyzing gesture...'):
            img_tensor = preprocess_image(image)
            
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
        
        predicted_letter = ASL_LETTERS[predicted.item()]
        confidence_pct = confidence.item() * 100
        
        # Prediction display
        st.markdown(f"""
        <div class='prediction-letter'>
            {predicted_letter}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='confidence-score'>
            Confidence: {confidence_pct:.2f}%
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(confidence_pct / 100)
        
        # Confidence interpretation
        if confidence_pct >= 95:
            st.markdown("""
            <div class='info-card'>
                <p style='margin: 0; color: #2C3E2A;'><strong>Very High Confidence</strong></p>
                <p style='margin: 0.25rem 0 0 0; color: #2C3E2A;'>The model is very confident about this prediction.</p>
            </div>
            """, unsafe_allow_html=True)
        elif confidence_pct >= 80:
            st.markdown("""
            <div class='info-card'>
                <p style='margin: 0; color: #2C3E2A;'><strong>Good Confidence</strong></p>
                <p style='margin: 0.25rem 0 0 0; color: #2C3E2A;'>This prediction is likely correct.</p>
            </div>
            """, unsafe_allow_html=True)
        elif confidence_pct >= 60:
            st.markdown("""
            <div class='info-card'>
                <p style='margin: 0; color: #2C3E2A;'><strong>Moderate Confidence</strong></p>
                <p style='margin: 0.25rem 0 0 0; color: #2C3E2A;'>Consider checking alternative predictions.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='info-card'>
                <p style='margin: 0; color: #2C3E2A;'><strong>Low Confidence</strong></p>
                <p style='margin: 0.25rem 0 0 0; color: #2C3E2A;'>Image may be unclear or ambiguous.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Top 3 predictions
        st.markdown("<h4 style='color: #2C3E2A;'>Top 3 Predictions</h4>", unsafe_allow_html=True)
        
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        
        for i in range(3):
            letter = ASL_LETTERS[top3_idx[0][i].item()]
            prob = top3_prob[0][i].item() * 100
            
            rank = ["1st", "2nd", "3rd"][i]
            
            st.markdown(f"""
            <div class='prediction-item'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='font-weight: 600; color: #2C3E2A; font-size: 1.1rem;'>{rank}: {letter}</span>
                    <span style='color: #2C3E2A; font-size: 1.1rem;'>{prob:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(prob / 100)
        
    else:
        st.markdown("""
        <div class='info-card'>
            <p style='margin: 0; color: #2C3E2A;'><strong>Waiting for image upload</strong></p>
            <p style='margin: 0.5rem 0 0 0; color: #2C3E2A;'>Upload an image to see predictions</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h4 style='color: #2C3E2A;'>Tips for Best Results</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div style='color: #2C3E2A;'>
        
        • Use clear, well-lit images<br>
        • Plain background preferred<br>
        • Center hand in frame<br>
        • Ensure fingers are visible
        
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #5F7359; padding: 2rem 0;'>
    <p>ASL Recognition System | Built with PyTorch & Streamlit</p>
</div>
""", unsafe_allow_html=True)