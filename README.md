# 🪐 Delphi Survey Data Processor - Space Edition 🌌

A stunning, space-themed Streamlit application for processing Delphi Survey data with planet animations and cosmic UI elements.

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## ✨ Features

### 🎨 Stunning Space Theme
- **Animated star background** with twinkling effects
- **Orbiting planet loading animations**
- **Cosmic color gradients** (purple, blue, pink)
- **Success animations** with galaxy effects
- **Smooth transitions** throughout

### ⚙️ Data Processing
- **S2_AGG Fields**: Weighted aggregates using RMSE weighting (70 columns)
- **IQR Fields**: Interquartile ranges using R Type 7 method (56 columns)
- **Peers Fields**: Individual peer response lists (70 columns)
- **Total**: 196 new columns generated

### 🔧 Advanced Features
- **Real-time validation** with detailed error reporting
- **Preview before download** with tabbed interface
- **Progress tracking** with animated planets
- **Verbose logging** option for debugging
- **Column group filtering** in preview
- **Memory usage tracking**
- **Automatic missing data handling**

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Quick Setup

1. **Clone or download the repository**
   ```bash
   # Navigate to your project directory
   cd delphi-survey-processor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import streamlit; print(f'Streamlit {streamlit.__version__} installed ✓')"
   ```

---

## 🚀 Usage

### Starting the Application

```bash
streamlit run cosmic_delphi_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Step-by-Step Guide

1. **📤 Upload Your CSV**
   - Click the file uploader
   - Select your Survey 1 CSV file
   - Wait for the cosmic data loading animation

2. **🔍 Validate Data**
   - Review validation metrics:
     - Number of respondents
     - Number of columns
     - Number of measures found
   - Check for warnings or errors

3. **🚀 Launch Processing**
   - Click the "Launch Processing Mission" button
   - Watch the planet animations while processing
   - Monitor progress bar and status updates

4. **👀 Preview Results**
   - Switch between tabs:
     - **All Data**: Complete dataset
     - **S2_AGG**: Aggregate statistics
     - **IQR**: Interquartile ranges
     - **Peers**: Individual peer lists
   - Adjust preview rows in sidebar

5. **💾 Download**
   - Click "Download Survey 2 Contact Data"
   - File saved as `{original_name}_Survey2_Data.csv`
   - Process new file or exit

---

## 📁 File Structure

```
delphi-survey-processor/
├── cosmic_delphi_app.py      # Main Streamlit app (space-themed)
├── delphi_processor.py        # Core processing module
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── data/                      # (Optional) Sample data folder
    └── Survey1_sample.csv
```

---

## 🔧 Configuration

### Sidebar Settings

- **Show validation details**: Toggle validation metrics display
- **Preview rows**: Adjust number of rows shown (5-50)
- **Verbose logging**: Enable detailed processing logs

### Processing Options

The app uses these defaults:
- Missing peer data: `Use default α=2, β=2`
- CDF sample points: `T=300`
- Beta fitting method: `L-BFGS-B`
- Quantile method: `R Type 7 (Hyndman & Fan 1996)`

---

## 📊 Input Data Requirements

### Required Columns per Measure

For each measure (e.g., `ARC-20`, `PGA_of _PsA`, etc.):

- `{Measure}_Range_Lower` (e.g., `ARC-20_Range_Lower`)
- `{Measure}_Range_Upper`
- `{Measure}_Mode`
- `{Measure}_StdDev`
- `{Measure}_Alpha`
- `{Measure}_Beta`

### Optional Columns

- `{Measure}_Quartile_1`
- `{Measure}_Quartile_3`
- `{Measure}_Median`
- `{Measure}_Mean`

### Supported Measures

1. ARC-20
2. PGA_of _PsA
3. MDA
4. Nail
5. Serious _inf
6. Upper_resp _inf
7. MACE

---

## 📈 Output Structure

### Column Groups (in order)

1. **Original Columns** (preserved exactly)
2. **S2_AGG_*** - Weighted aggregates
   - Range_Lower, Range_Upper
   - Mode, Mean, Median
   - Quartile_1, Quartile_3
   - StdDev, Alpha, Beta
3. **IQR_*** - Interquartile ranges
   - Range_Lower_min, Range_Lower_max
   - Range_Upper_min, Range_Upper_max
   - Mode_min, Mode_max
   - StdDev_min, StdDev_max
4. **Peers_*** - Peer lists (CSV format)
   - All 10 statistics as comma-separated values

### Example Output Columns

```
Original columns...
S2_AGG_ARC-20_Range_Lower
S2_AGG_ARC-20_Range_Upper
S2_AGG_ARC-20_Mode
...
IQR_arc20_Range_Lower_min
IQR_arc20_Range_Lower_max
...
Peers_ARC-20_Range_Lower
Peers_ARC-20_Range_Upper
...
```

---

## 🎨 UI Customization

### Color Scheme

The app uses a cosmic color palette:
- **Primary**: `#667eea` (Purple)
- **Secondary**: `#764ba2` (Deep Purple)
- **Accent**: `#f093fb` (Pink)
- **Background**: `#0a0e27` (Dark Blue)
- **Text**: `#e0e6ff` (Light Blue)

### Animations

- **Stars**: Twinkling background (100 stars)
- **Planets**: Orbiting animation (3 planets)
- **Success**: Spinning star with pulse effect
- **Hover**: Lift and glow effects on cards
- **Progress**: Gradient progress bar

---

## 🔬 Technical Details

### Processing Algorithm

1. **Validation Phase**
   - Check for required columns
   - Validate data types
   - Count available measures

2. **Aggregation Phase** (per respondent, per measure)
   - Extract peer data (exclude current respondent)
   - Compute CDF matrix (300 sample points)
   - Calculate MSE for each peer
   - Compute inverse RMSE weights
   - Generate weighted aggregate CDF
   - Fit Beta distribution using L-BFGS-B

3. **IQR Calculation**
   - Use R Type 7 quantile method
   - Calculate 25th and 75th percentiles
   - Apply to Range_Lower, Range_Upper, Mode, StdDev

4. **Peer Lists**
   - Store all peer values as CSV strings
   - Include all 10 statistics per measure

### Performance

- **Speed**: ~1-2 seconds per respondent (depends on #measures)
- **Memory**: Efficient pandas operations
- **Scalability**: Tested with 100+ respondents

---

## 🐛 Troubleshooting

### Common Issues

**Problem**: "No valid measures found"
- **Solution**: Ensure column names match exactly (case-sensitive)
- Check for typos in measure names

**Problem**: App won't start
- **Solution**: Verify Python version (3.9+)
- Check all dependencies installed: `pip list`

**Problem**: Processing fails
- **Solution**: Check for missing required columns
- Enable verbose logging to see detailed errors

**Problem**: Download button not working
- **Solution**: Ensure processing completed successfully
- Check browser download settings

### Debug Mode

Enable verbose logging in sidebar to see:
- Processing progress per respondent
- Measure-by-measure status
- Error messages with context

---

## 📝 Code Architecture

### Module: `delphi_processor.py`

**Core Functions:**
- `process_survey_data()` - Main processing pipeline
- `validate_input_data()` - Data validation
- `compute_aggregate_cdf_from_peers()` - Aggregation
- `r_quantile_type7()` - IQR calculation
- `fit_beta_to_cdf()` - Beta distribution fitting

**Helper Functions:**
- `normalize_token()` - String normalization
- `join_vals()` - Array to CSV conversion
- `beta_quantile()`, `beta_mean()` - Beta statistics
- `mode_sd_from_alpha_beta()` - Parameter conversion

### App: `cosmic_delphi_app.py`

**UI Components:**
- `render_header()` - Cosmic header
- `render_sidebar()` - Mission control sidebar
- `render_stars()` - Animated background
- `show_planet_loading()` - Loading animation
- `show_success_animation()` - Success feedback
- `render_metric_card()` - Animated metrics

**Main Flow:**
- File upload → Validation → Processing → Preview → Download

---

## 📚 References

### Delphi Methods Specification
- Aggregation: Section 5.1.3 (weighted RMSE)
- IQR: Section 5.3 (R Type 7 quantiles)
- Beta fitting: L-BFGS-B optimization
- Mode equation: (α-1)/(α+β-2)
- Variance: αβ/((α+β)²(α+β+1))

### Academic References
- Hyndman, R.J. and Fan, Y. (1996). Sample Quantiles in Statistical Packages. *The American Statistician*, 50(4), 361-365.

---

## 🤝 Support

### Getting Help

1. Check the **Mission Brief** in the app
2. Review validation warnings/errors
3. Enable verbose logging for details
4. Check this README for troubleshooting

### Reporting Issues

When reporting issues, include:
- Error message (full text)
- Input file structure (column names)
- Number of respondents
- Streamlit version: `streamlit --version`
- Python version: `python --version`

---

## 🎯 Future Enhancements

- [ ] Export to Excel with multiple sheets
- [ ] Interactive data visualization dashboard
- [ ] Batch processing for multiple files
- [ ] Custom measure configuration
- [ ] Advanced filtering options
- [ ] Data quality reports
- [ ] Export processing logs

---

## 📄 License

MIT License - Feel free to use and modify for your needs.

---

## 🌟 Credits

**Developer**: Data Engineering Team  
**Version**: 2.0.0 - Space Edition  
**Based on**: Delphi Methods (Jan 2026)  
**UI Inspiration**: Cosmos, planets, and the beauty of space 🌌

---

## 🚀 Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run cosmic_delphi_app.py

# Run with custom port
streamlit run cosmic_delphi_app.py --server.port 8502

# Run with custom browser
streamlit run cosmic_delphi_app.py --browser.serverAddress localhost
```

---

**Enjoy your cosmic data processing journey! 🪐✨**
