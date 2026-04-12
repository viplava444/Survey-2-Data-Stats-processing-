#!/usr/bin/env python3
"""
🪐 Delphi Survey Data Processor - Space Edition 🌌
==================================================

A stunning, space-themed Streamlit app for processing Delphi Survey data
with planet animations and cosmic UI elements.

Author: Data Engineering Team
Version: 2.0.0 - Space Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import time
from pathlib import Path

# Import processing functions
from delphi_processor import (
    process_survey_data,
    validate_input_data,
    get_required_columns,
    MEASURES
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🪐 Delphi Survey Processor",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS WITH SPACE THEME
# ============================================================================

st.markdown("""
<style>
    /* Space Background */
    .stApp {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
        color: #e0e6ff;
    }
    
    /* Animated Stars Background */
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    
    .stars {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
    }
    
    .star {
        position: absolute;
        width: 2px;
        height: 2px;
        background: white;
        border-radius: 50%;
        animation: twinkle 3s infinite;
    }
    
    /* Main Header */
    .cosmic-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 2rem 0;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: brightness(1); }
        to { filter: brightness(1.3); }
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #a5b4fc;
        text-align: center;
        margin-bottom: 3rem;
        opacity: 0.9;
    }
    
    /* Planet Loading Animation */
    .planet-loader {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 3rem 0;
    }
    
    .planet {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        position: relative;
        animation: orbit 3s linear infinite;
    }
    
    .planet-1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 0 40px rgba(102, 126, 234, 0.6);
    }
    
    .planet-2 {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 0 40px rgba(245, 87, 108, 0.6);
        margin: 0 2rem;
    }
    
    .planet-3 {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 0 40px rgba(79, 172, 254, 0.6);
    }
    
    @keyframes orbit {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
        100% { transform: translateY(0px) rotate(360deg); }
    }
    
    /* Success Animation */
    .success-galaxy {
        text-align: center;
        padding: 2rem;
        margin: 2rem 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 20px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); box-shadow: 0 0 20px rgba(102, 126, 234, 0.3); }
        50% { transform: scale(1.02); box-shadow: 0 0 40px rgba(102, 126, 234, 0.6); }
    }
    
    .success-emoji {
        font-size: 4rem;
        animation: spin 1s ease-in-out;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg) scale(0); }
        50% { transform: rotate(180deg) scale(1.2); }
        100% { transform: rotate(360deg) scale(1); }
    }
    
    /* Cards */
    .cosmic-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a5b4fc;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(245, 87, 108, 0.4);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.6);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    
    /* File Uploader */
    .uploadedFile {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        border: 2px dashed rgba(102, 126, 234, 0.3);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 10px;
        color: #a5b4fc;
    }
    
    /* Dataframe */
    .dataframe {
        background: rgba(10, 14, 39, 0.5);
        border-radius: 10px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        color: #a5b4fc;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ANIMATED BACKGROUND
# ============================================================================

def render_stars():
    """Render animated star background."""
    stars_html = '<div class="stars">'
    np.random.seed(42)
    for _ in range(100):
        x = np.random.randint(0, 100)
        y = np.random.randint(0, 100)
        delay = np.random.uniform(0, 3)
        stars_html += f'<div class="star" style="left: {x}%; top: {y}%; animation-delay: {delay}s;"></div>'
    stars_html += '</div>'
    st.markdown(stars_html, unsafe_allow_html=True)

# ============================================================================
# PLANET LOADING ANIMATION
# ============================================================================

def show_planet_loading(message="Processing through the cosmos..."):
    """Display animated planet loading indicator."""
    st.markdown(f"""
    <div class="planet-loader">
        <div class="planet planet-1"></div>
        <div class="planet planet-2"></div>
        <div class="planet planet-3"></div>
    </div>
    <p style="text-align: center; color: #a5b4fc; font-size: 1.2rem; margin-top: 1rem;">
        {message}
    </p>
    """, unsafe_allow_html=True)

def show_success_animation(message="Mission Complete! 🎉"):
    """Display success animation."""
    st.markdown(f"""
    <div class="success-galaxy">
        <div class="success-emoji">🌟</div>
        <h2 style="color: #667eea; margin-top: 1rem;">{message}</h2>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def render_header():
    """Render cosmic header."""
    st.markdown(
        '<div class="cosmic-header">🪐 Delphi Survey Processor 🌌</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Transform Survey Round 1 responses into Survey Round 2 contact data</div>',
        unsafe_allow_html=True
    )

def render_metric_card(value, label, col):
    """Render animated metric card."""
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

def render_sidebar():
    """Render cosmic sidebar."""
    with st.sidebar:
        st.markdown("### 🌠 Mission Control")
        
        st.markdown("""
        <div class="cosmic-card">
            <h4 style="color: #667eea;">📋 Process Overview</h4>
            <ol style="color: #a5b4fc;">
                <li>Upload CSV file 📤</li>
                <li>Validate data structure ✓</li>
                <li>Process aggregations ⚙️</li>
                <li>Preview results 👀</li>
                <li>Download output 💾</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown(f"""
        <div class="cosmic-card">
            <h4 style="color: #667eea;">🎯 Generated Fields</h4>
            <ul style="color: #a5b4fc; list-style: none; padding: 0;">
                <li>📊 <b>S2_AGG</b>: {len(MEASURES)} × 10 = 70 cols</li>
                <li>📈 <b>IQR</b>: {len(MEASURES)} × 8 = 56 cols</li>
                <li>👥 <b>Peers</b>: {len(MEASURES)} × 10 = 70 cols</li>
            </ul>
            <hr style="border-color: rgba(102, 126, 234, 0.2);">
            <p style="text-align: center; color: #667eea; font-size: 1.2rem; font-weight: bold;">
                Total: 196 new columns
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.expander("⚙️ Settings"):
            show_validation = st.checkbox("Show validation details", value=True)
            show_preview_rows = st.slider("Preview rows", 5, 50, 10)
            verbose_logging = st.checkbox("Verbose logging", value=False)
        
        st.markdown("---")
        
        with st.expander("📖 Supported Measures"):
            for i, measure in enumerate(MEASURES, 1):
                st.markdown(f"**{i}.** {measure}")
        
        return show_validation, show_preview_rows, verbose_logging

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application logic."""
    
    # Render background and UI
    render_stars()
    render_header()
    show_validation, show_preview_rows, verbose_logging = render_sidebar()
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    
    # File upload section
    st.markdown("""
    <div class="cosmic-card">
        <h3 style="color: #667eea;">📤 Upload Survey 1 Data</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your CSV file from the cosmos",
        type=['csv'],
        help="Upload Survey Round 1 response data in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            loading_container = st.empty()
            with loading_container.container():
                show_planet_loading("Reading data from the stars...")
                time.sleep(1)
            
            df_input = pd.read_csv(uploaded_file)
            loading_container.empty()
            
            st.success(f"✨ File loaded: **{uploaded_file.name}**")
            
            # Input data preview
            with st.expander("🔭 Input Data Overview", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                render_metric_card(df_input.shape[0], "Rows", col1)
                render_metric_card(df_input.shape[1], "Columns", col2)
                memory_mb = df_input.memory_usage(deep=True).sum() / 1024**2
                render_metric_card(f"{memory_mb:.1f} MB", "Memory", col3)
                missing = df_input.isnull().sum().sum()
                render_metric_card(missing, "Missing", col4)
                
                st.dataframe(df_input.head(show_preview_rows), use_container_width=True)
            
            # Validation
            st.markdown("""
            <div class="cosmic-card">
                <h3 style="color: #667eea;">🔍 Data Validation</h3>
            </div>
            """, unsafe_allow_html=True)
            
            validation_results = validate_input_data(df_input)
            
            if show_validation:
                col1, col2, col3 = st.columns(3)
                render_metric_card(validation_results['n_rows'], "Respondents", col1)
                render_metric_card(validation_results['n_cols'], "Columns", col2)
                render_metric_card(validation_results['n_measures'], "Measures", col3)
                
                if validation_results['warnings']:
                    for warning in validation_results['warnings']:
                        st.warning(f"⚠️ {warning}")
                
                if validation_results['errors']:
                    for error in validation_results['errors']:
                        st.error(f"❌ {error}")
            
            if not validation_results['is_valid']:
                st.stop()
            
            # Process button
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Launch Processing Mission", type="primary", use_container_width=True):
                    st.session_state.processing = True
            
            # Processing
            if st.session_state.get('processing', False):
                progress_container = st.empty()
                log_container = st.empty()
                
                try:
                    # Show loading animation
                    with progress_container.container():
                        show_planet_loading("Processing through the cosmos...")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                    
                    logs = []
                    
                    def log_callback(message):
                        logs.append(message)
                        if verbose_logging:
                            with log_container.expander("📝 Mission Log", expanded=True):
                                st.text("\n".join(logs[-15:]))
                    
                    # Simulate progress
                    for i in range(0, 30, 5):
                        progress_bar.progress(i)
                        time.sleep(0.1)
                    
                    status_text.text("🌌 Computing aggregations...")
                    progress_bar.progress(40)
                    
                    # Process data
                    df_output = process_survey_data(
                        df_input,
                        handle_missing="Use default α=2, β=2",
                        verbose=verbose_logging,
                        callback=log_callback if verbose_logging else None
                    )
                    
                    progress_bar.progress(90)
                    status_text.text("✨ Finalizing cosmic data...")
                    time.sleep(0.5)
                    
                    # Store results
                    st.session_state.processed_data = df_output
                    st.session_state.input_filename = uploaded_file.name
                    st.session_state.processed = True
                    st.session_state.processing = False
                    
                    progress_bar.progress(100)
                    time.sleep(0.3)
                    progress_container.empty()
                    
                    # Success animation
                    show_success_animation("🎉 Mission Accomplished!")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    progress_container.empty()
                    st.error(f"❌ Mission failed: {str(e)}")
                    st.exception(e)
                    st.session_state.processing = False
                    return
            
            # Display results
            if st.session_state.processed and 'processed_data' in st.session_state:
                df_output = st.session_state.processed_data
                
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("""
                <div class="cosmic-card">
                    <h3 style="color: #667eea;">✅ Processing Complete!</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                new_cols = df_output.shape[1] - df_input.shape[1]
                render_metric_card(df_input.shape[1], "Original Cols", col1)
                render_metric_card(f"+{new_cols}", "New Cols", col2)
                render_metric_card(df_output.shape[1], "Total Cols", col3)
                output_mb = df_output.memory_usage(deep=True).sum() / 1024**2
                render_metric_card(f"{output_mb:.1f} MB", "Output Size", col4)
                
                # Column breakdown
                with st.expander("📊 New Columns Breakdown"):
                    agg_cols = [c for c in df_output.columns if c.startswith('S2_AGG_')]
                    iqr_cols = [c for c in df_output.columns if c.startswith('IQR_')]
                    peer_cols = [c for c in df_output.columns if c.startswith('Peers_')]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="cosmic-card" style="text-align: center;">
                            <h4 style="color: #667eea;">S2_AGG</h4>
                            <p style="font-size: 2rem; color: #667eea;">{len(agg_cols)}</p>
                            <p style="color: #a5b4fc;">Aggregate fields</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="cosmic-card" style="text-align: center;">
                            <h4 style="color: #764ba2;">IQR</h4>
                            <p style="font-size: 2rem; color: #764ba2;">{len(iqr_cols)}</p>
                            <p style="color: #a5b4fc;">Quartile ranges</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="cosmic-card" style="text-align: center;">
                            <h4 style="color: #f093fb;">Peers</h4>
                            <p style="font-size: 2rem; color: #f093fb;">{len(peer_cols)}</p>
                            <p style="color: #a5b4fc;">Peer lists</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Preview tabs
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                <div class="cosmic-card">
                    <h3 style="color: #667eea;">👀 Preview Processed Data</h3>
                </div>
                """, unsafe_allow_html=True)
                
                tab1, tab2, tab3, tab4 = st.tabs(["📊 All Data", "📈 S2_AGG", "📉 IQR", "👥 Peers"])
                
                with tab1:
                    st.dataframe(df_output.head(show_preview_rows), use_container_width=True)
                
                with tab2:
                    if agg_cols:
                        st.dataframe(df_output[agg_cols].head(show_preview_rows), use_container_width=True)
                
                with tab3:
                    if iqr_cols:
                        st.dataframe(df_output[iqr_cols].head(show_preview_rows), use_container_width=True)
                
                with tab4:
                    if peer_cols:
                        st.dataframe(df_output[peer_cols].head(show_preview_rows), use_container_width=True)
                
                # Download section
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                <div class="cosmic-card">
                    <h3 style="color: #667eea;">💾 Download Your Data</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Convert to CSV
                csv_buffer = io.StringIO()
                df_output.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                # Output filename
                input_name = st.session_state.input_filename
                output_name = input_name.replace('.csv', '_Survey2_Data.csv')
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    st.download_button(
                        label="🌟 Download Survey 2 Contact Data",
                        data=csv_data,
                        file_name=output_name,
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )
                    
                    if st.button("🔄 Process New File", use_container_width=True):
                        st.session_state.clear()
                        st.rerun()
        
        except pd.errors.EmptyDataError:
            st.error("❌ The uploaded file is empty. Please check your data.")
        except pd.errors.ParserError:
            st.error("❌ Failed to parse CSV file. Please verify the file format.")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.exception(e)
    
    else:
        # No file uploaded - show instructions
        st.markdown("""
        <div class="cosmic-card" style="text-align: center; padding: 3rem;">
            <h2 style="color: #667eea;">🌌 Welcome, Space Explorer!</h2>
            <p style="color: #a5b4fc; font-size: 1.2rem; margin: 2rem 0;">
                Upload your Survey 1 CSV file to begin the cosmic transformation
            </p>
            <p style="color: #667eea; font-size: 3rem;">☝️</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📚 Mission Brief"):
            st.markdown(f"""
            ### 🎯 Mission Objectives
            
            Transform Survey Round 1 data into Survey Round 2 contact data with:
            
            1. **S2_AGG Fields** - Weighted aggregates using RMSE weighting
            2. **IQR Fields** - Interquartile ranges using R Type 7 method
            3. **Peers Fields** - Individual peer response lists
            
            ### 📋 Supported Measures
            """)
            
            cols = st.columns(2)
            for i, measure in enumerate(MEASURES):
                with cols[i % 2]:
                    st.markdown(f"✨ **{measure}**")
            
            st.markdown("""
            ### 🚀 Launch Sequence
            
            1. Prepare your Survey 1 CSV with required columns
            2. Upload the file using the uploader above
            3. Review validation results
            4. Click "Launch Processing Mission"
            5. Preview the transformed data
            6. Download your Survey 2 contact data
            
            ### ⚡ Pro Tips
            
            - Enable verbose logging for detailed mission logs
            - Adjust preview rows to see more/less data
            - All original columns are preserved
            - Missing data is handled automatically
            """)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
