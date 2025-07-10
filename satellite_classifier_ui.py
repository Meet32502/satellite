import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="🛰️ Satellite Land Detection",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Model configuration
MODEL_URL = "https://drive.google.com/uc?export=download&id=1p9pqC-Ba4aKdNcQploHjnaCVip5J07qe"
MODEL_PATH = "Modelenv.v1.h5"

# Download model if not present
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🚀 Downloading AI model... (This happens only once)"):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    return load_model(MODEL_PATH)

# Class information
class_info = {
    'Cloudy': {
        'color': '#87CEEB',
        'icon': '☁️',
        'description': 'Cloud formations and overcast areas'
    },
    'Desert': {
        'color': '#F4A460',
        'icon': '🏜️',
        'description': 'Arid lands and sandy regions'
    },
    'Green_Area': {
        'color': '#32CD32',
        'icon': '🌱',
        'description': 'Vegetation and forested areas'
    },
    'Water': {
        'color': '#4682B4',
        'icon': '💧',
        'description': 'Water bodies and aquatic areas'
    }
}

class_names = list(class_info.keys())

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# Header
st.markdown('<h1 class="main-header">🛰️ Satellite Land Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-powered land classification from satellite imagery</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Dashboard")
    
    # Model status
    try:
        model = download_and_load_model()
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
    
    # Statistics
    if st.session_state.predictions_history:
        st.subheader("📈 Session Statistics")
        total_predictions = len(st.session_state.predictions_history)
        st.metric("Total Predictions", total_predictions)
        
        # Most common prediction
        predictions_df = pd.DataFrame(st.session_state.predictions_history)
        most_common = predictions_df['predicted_class'].mode()[0]
        st.metric("Most Common Class", most_common)
        
        # Average confidence
        avg_confidence = predictions_df['confidence'].mean()
        st.metric("Average Confidence", f"{avg_confidence:.1f}%")
    
    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.predictions_history = []
        st.rerun()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Satellite Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a satellite image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a satellite image to classify land type"
    )
    
    # Sample images section
    st.subheader("🖼️ Or try sample images")
    sample_images = {
        "Cloudy": "https://via.placeholder.com/150/87CEEB/000000?text=Cloudy",
        "Desert": "https://via.placeholder.com/150/F4A460/000000?text=Desert",
        "Green Area": "https://via.placeholder.com/150/32CD32/000000?text=Green",
        "Water": "https://via.placeholder.com/150/4682B4/000000?text=Water"
    }
    
    sample_cols = st.columns(2)
    for i, (name, url) in enumerate(sample_images.items()):
        with sample_cols[i % 2]:
            if st.button(f"Try {name}", key=f"sample_{name}"):
                st.info(f"Sample {name} image selected (placeholder)")

with col2:
    st.subheader("🔍 Analysis Results")
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        original_size = image.size
        image_resized = image.resize((256, 256))
        
        st.image(image, caption="Uploaded Satellite Image", use_container_width=True)
        
        # Image info
        st.caption(f"Original size: {original_size[0]}x{original_size[1]} pixels")
        
        # Prediction
        with st.spinner("🤖 Analyzing image..."):
            # Simulate processing time for better UX
            time.sleep(1)
            
            # Preprocess
            img_array = img_to_array(image_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = model.predict(img_array)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            # Store prediction
            st.session_state.predictions_history.append({
                'timestamp': datetime.now(),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'filename': uploaded_file.name
            })
        
        # Display results
        st.success(f"🎯 **Prediction: {class_info[predicted_class]['icon']} {predicted_class}**")
        st.info(f"📊 **Confidence: {confidence:.2f}%**")
        
        # Confidence bar
        confidence_normalized = confidence / 100
        st.progress(float(confidence_normalized))
        
        # Class description
        st.markdown(f"**Description:** {class_info[predicted_class]['description']}")
        
    else:
        st.info("👆 Please upload a satellite image to begin analysis")

# Visualization section
if uploaded_file is not None:
    st.divider()
    st.subheader("📊 Detailed Analysis")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Confidence chart for all classes
        st.subheader("🎯 Class Confidence Scores")
        
        confidence_data = pd.DataFrame({
            'Class': class_names,
            'Confidence': prediction * 100,
            'Color': [class_info[name]['color'] for name in class_names]
        })
        
        fig_bar = px.bar(
            confidence_data, 
            x='Class', 
            y='Confidence',
            color='Color',
            color_discrete_map=dict(zip(confidence_data['Color'], confidence_data['Color'])),
            title="Confidence Score by Land Type"
        )
        fig_bar.update_layout(
            showlegend=False,
            xaxis_title="Land Type",
            yaxis_title="Confidence (%)",
            height=400
        )
        fig_bar.update_traces(
            texttemplate='%{y:.1f}%',
            textposition='outside'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with viz_col2:
        # Pie chart of confidence distribution
        st.subheader("🥧 Confidence Distribution")
        
        fig_pie = px.pie(
            confidence_data,
            values='Confidence',
            names='Class',
            color='Class',
            color_discrete_map={name: info['color'] for name, info in class_info.items()},
            title="Distribution of Confidence Scores"
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

# History section
if st.session_state.predictions_history:
    st.divider()
    st.subheader("📚 Prediction History")
    
    history_df = pd.DataFrame(st.session_state.predictions_history)
    
    # Recent predictions table
    st.subheader("🕐 Recent Predictions")
    display_df = history_df.copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
    display_df = display_df.rename(columns={
        'timestamp': 'Time',
        'predicted_class': 'Predicted Class',
        'confidence': 'Confidence (%)',
        'filename': 'Filename'
    })
    st.dataframe(display_df, use_container_width=True)
    
    # History visualizations
    if len(history_df) > 1:
        hist_col1, hist_col2 = st.columns(2)
        
        with hist_col1:
            # Confidence over time
            st.subheader("📈 Confidence Over Time")
            fig_line = px.line(
                history_df, 
                x='timestamp', 
                y='confidence',
                title="Confidence Scores Over Time",
                markers=True
            )
            fig_line.update_layout(
                xaxis_title="Time",
                yaxis_title="Confidence (%)",
                height=300
            )
            st.plotly_chart(fig_line, use_container_width=True)
        
        with hist_col2:
            # Class distribution
            st.subheader("📊 Class Distribution")
            class_counts = history_df['predicted_class'].value_counts()
            
            fig_donut = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title="Distribution of Predicted Classes",
                hole=0.4,
                color=class_counts.index,
                color_discrete_map={name: info['color'] for name, info in class_info.items()}
            )
            fig_donut.update_layout(height=300)
            st.plotly_chart(fig_donut, use_container_width=True)

# Footer
st.divider()
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🛰️ About")
    st.markdown("This AI system classifies satellite images into four main land types using deep learning.")

with col2:
    st.markdown("### 📊 Supported Classes")
    for name, info in class_info.items():
        st.markdown(f"{info['icon']} **{name}**: {info['description']}")

with col3:
    st.markdown("### 🎯 Model Info")
    st.markdown("- **Architecture**: Convolutional Neural Network")
    st.markdown("- **Input Size**: 256x256 pixels")
    st.markdown("- **Classes**: 4 land types")
    st.markdown(f"- **Status**: {'✅ Active' if 'model' in locals() else '❌ Inactive'}")

# Performance tips
with st.expander("💡 Tips for Better Results"):
    st.markdown("""
    - **Image Quality**: Use high-resolution satellite images for better accuracy
    - **Clear Views**: Avoid images with excessive cloud cover (unless classifying clouds)
    - **Proper Lighting**: Daylight images generally work better than nighttime
    - **File Format**: JPG and PNG formats are supported
    - **Size**: Images are automatically resized to 256x256 pixels for processing
    """)

# Technical details
with st.expander("🔧 Technical Details"):
    st.markdown("""
    - **Framework**: TensorFlow/Keras
    - **Model Type**: Convolutional Neural Network (CNN)
    - **Input Processing**: Images normalized to [0,1] range
    - **Output**: Softmax probabilities for each class
    - **Confidence**: Maximum probability score
    """)
