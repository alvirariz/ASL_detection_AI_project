import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# ===== MODEL ARCHITECTURE =====
# Copy the exact architecture from Person 2's code
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
# Order: 0-25 = A-Z, 26 = del, 27 = nothing, 28 = space
ASL_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
               'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# ===== LOAD MODEL =====
@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    model = DeepCNN(dropout_rate=0.5, dense_units=256)
    model.load_state_dict(torch.load('model_v3.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# ===== PREPROCESS IMAGE =====
def preprocess_image(image):
    """
    Preprocess uploaded image to match training data format:
    1. Convert to grayscale
    2. Resize to 64x64
    3. Normalize (divide by 255)
    4. Reshape for model input
    """
    # Convert to grayscale
    img = image.convert('L')
    
    # Resize to 64x64
    img = img.resize((64, 64))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Reshape to (1, 1, 64, 64) - batch_size=1, channels=1, height=64, width=64
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    return img_tensor

# ===== STREAMLIT APP UI =====

# Page configuration
st.set_page_config(
    page_title="ASL Sign Language Recognizer",
    page_icon="🤟",
    layout="wide"
)

# Title and description
st.title("🤟 ASL Sign Language Recognition")
st.markdown("**Upload an image of an ASL hand gesture and get instant predictions!**")
st.markdown("---")

# Sidebar with information
with st.sidebar:
    st.header("📊 Model Information")
    st.markdown("""
    **Architecture:** Deep CNN  
    **Model:** V3 (Winner)  
    **Validation Accuracy:** 99.90%  
    **Classes:** 29 ASL letters  
    **Hyperparameters:**
    - Dropout: 0.5
    - Dense Units: 256
    - Optimizer: Adam
    - Learning Rate: 0.001
    """)
    
    st.markdown("---")
    
    st.header("📖 How to Use")
    st.markdown("""
    1. Upload an ASL hand gesture image
    2. Wait for the model to process
    3. View the predicted letter
    4. Check confidence score
    5. See top 3 predictions
    """)
    
    st.markdown("---")
    
    st.header("ℹ️ About ASL")
    st.markdown("""
    American Sign Language (ASL) uses hand shapes, movements, and facial expressions to communicate. This model recognizes static hand gestures for letters A-Z plus special characters.
    """)

# Main content area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Show image info
        st.caption(f"Image size: {image.size[0]} × {image.size[1]} pixels")
    else:
        # Placeholder when no image uploaded
        st.info("👆 Click 'Browse files' to upload an ASL gesture image")
        st.image("https://via.placeholder.com/400x400.png?text=Upload+ASL+Gesture", 
                 caption="Waiting for upload...", 
                 use_container_width=True)

with col2:
    st.subheader("🔮 Prediction Results")
    
    if uploaded_file is not None:
        # Load model with spinner
        with st.spinner('🔄 Loading model...'):
            model = load_model()
        
        # Make prediction with spinner
        with st.spinner('🧠 Analyzing hand gesture...'):
            # Preprocess image
            img_tensor = preprocess_image(image)
            
            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
        
        # Extract results
        predicted_letter = ASL_LETTERS[predicted.item()]
        confidence_pct = confidence.item() * 100
        
        # Display main prediction
        st.success("✅ Prediction Complete!")
        
        # Large prediction display
        st.markdown(f"## Predicted Letter: **:blue[{predicted_letter}]**")
        st.markdown(f"### Confidence: **{confidence_pct:.2f}%**")
        
        # Confidence progress bar
        st.progress(confidence_pct / 100)
        
        # Interpretation of confidence
        if confidence_pct >= 95:
            st.success("🎯 Very high confidence! The model is very sure.")
        elif confidence_pct >= 80:
            st.info("✓ Good confidence. Prediction is likely correct.")
        elif confidence_pct >= 60:
            st.warning("⚠️ Moderate confidence. Check alternatives below.")
        else:
            st.error("❌ Low confidence. Image may be unclear or ambiguous.")
        
        # Divider
        st.markdown("---")
        
        # Top 3 predictions table
        st.markdown("#### 🏆 Top 3 Predictions:")
        
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        
        # Create a nice table display
        for i in range(3):
            letter = ASL_LETTERS[top3_idx[0][i].item()]
            prob = top3_prob[0][i].item() * 100
            
            # Medal emojis for top 3
            medals = ["🥇", "🥈", "🥉"]
            
            col_medal, col_letter, col_bar = st.columns([0.5, 1, 3])
            
            with col_medal:
                st.markdown(f"### {medals[i]}")
            
            with col_letter:
                st.markdown(f"### **{letter}**")
            
            with col_bar:
                st.markdown(f"**{prob:.2f}%**")
                st.progress(prob / 100)
        
    else:
        # Placeholder when no image uploaded
        st.info("📸 Upload an image to see predictions")
        st.markdown("""
        **Tips for best results:**
        - Use clear, well-lit images
        - Plain background works best
        - Hand should be centered in frame
        - Fingers clearly visible
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with PyTorch & Streamlit | ASL Recognition Project</p>
</div>
""", unsafe_allow_html=True)