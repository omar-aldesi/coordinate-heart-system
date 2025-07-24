# Coordinate Heart System: A Geometric Framework for Emotion Representation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Paper](https://img.shields.io/badge/Research-Paper-red.svg)](https://arxiv.org/abs/2507.14593)

## 📚 Overview

The **Coordinate Heart System (CHS)** is a novel geometric framework for representing and analyzing human emotions in a continuous coordinate space. This implementation provides a computational model based on the research paper **"Coordinate Heart System: A Geometric Framework for Emotion Representation"** by Omar Aldesi.

### Key Features

- **Geometric Emotion Mapping**: Maps emotions to a continuous 2D coordinate system
- **Multi-dimensional Analysis**: Incorporates emotion intensities, stability metrics, and contextual factors
- **Real-time Processing**: Efficient analysis with detailed performance metrics
- **Academic Research Tool**: Designed for emotion recognition research and psychological studies
- **Extensible Framework**: Modular design for easy integration and customization

## 🎯 Academic Context

This implementation serves as a practical tool for researchers, students, and practitioners working in:

- **Affective Computing**: Computational models of human emotions
- **Psychology Research**: Quantitative emotion analysis
- **Human-Computer Interaction**: Emotion-aware systems
- **Natural Language Processing**: Sentiment and emotion detection
- **Machine Learning**: Feature engineering for emotion classification

## 📖 Research Paper

**Citation**: 
```
Aldesi, O. (2025). Coordinate Heart System: A Geometric Framework for Emotion Representation. 
[arXiv]. [arXiv:2507.14593](https://arxiv.org/abs/2507.14593)
```

**Paper Link**: [arXiv:2507.14593](https://arxiv.org/abs/2507.14593)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)
- Virtual environment tool (recommended)

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/omar-aldesi/coordinate-heart-system.git
cd coordinate-heart-system
```

#### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv chs_env
source chs_env/bin/activate  # On Windows: chs_env\Scripts\activate

# Or using conda
conda create -n chs_env python=3.8
conda activate chs_env
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure Environment Variables

The system requires API keys for LLM integration. Set up your environment:

```bash
# Copy the example environment file
cp .env.example .env
```

Edit the `.env` file with your API keys:

```bash
# Required: At least one LLM provider API key
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent
GEMINI_MODEL=gemini-2.5-flash

ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_BASE_URL=https://api.anthropic.com/v1/messages
ANTHROPIC_MODEL=claude-sonnet-4-20250514

OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1/chat/completions
OPENAI_MODEL=gpt-4o-mini
```

**⚠️ Important Security Notes:**
- Never commit your `.env` file to version control
- The `.env` file is already included in `.gitignore`
- Keep your API keys secure and private
- For production use, consider using environment variable management tools

#### 5. Obtain API Keys

Choose at least one LLM provider:

**Google Gemini** (Recommended for research):
- Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
- Create a new API key
- Free tier available for research use

**Anthropic Claude**:
- Sign up at [Anthropic Console](https://console.anthropic.com/)
- Generate API key in your dashboard
- Academic discounts may be available

**OpenAI GPT**:
- Create account at [OpenAI Platform](https://platform.openai.com/)
- Navigate to API keys section
- Note: Paid service, credits required

#### 6. Run the Example

```bash
python main.py
```

### Expected Output

The system will analyze the provided text and display:

```
🌟 Emotional Analysis System
🕐 Started at: 2025-07-19 13:52:01

🔍 Analyzing text: 'i usually show the typical in public, the uniqe when iam alone'
⏳ Processing...
🤖 Getting LLM response...
🧠 Running emotion analysis...

============================================================
🧠 EMOTIONAL ANALYSIS RESULTS
============================================================
📅 Analysis Time: 2025-07-19T13:52:01.527276
🎯 Dominant Emotion: FEAR
🏷️  Emotion Label: Dread (Unstable/Struggling)
📍 Coordinates: (0.372, -0.600)
⚖️  Stability: 0.300

----------------------------------------
💫 EMOTION INTENSITIES
----------------------------------------
    Fear: ██████░░░░░░░░░░░░░░ 0.300
    Pride: ██░░░░░░░░░░░░░░░░░░ 0.100
  Sadness: ████░░░░░░░░░░░░░░░░ 0.200

⏱️  PERFORMANCE BREAKDOWN
------------------------------
🤖 LLM Response Time: 1.245 seconds
🧠 CHS Analysis Time: 0.089 seconds
📊 Total Time: 1.334 seconds
```

## 🏗️ System Architecture

### Core Components

1. **CoordinateHeartSystem**: Main analysis engine that processes emotional data
2. **EmotionalState**: Data structure representing complete emotion analysis results
3. **EmotionIntensities**: Quantified intensities for eight primary emotions
4. **ContextualDrain**: Analysis of contextual factors affecting emotional state
5. **LLM Integration**: Natural language processing for emotion extraction

### Emotion Categories

The system analyzes eight primary emotions:
- **Joy**: Positive, high-energy emotions
- **Anger**: Negative, high-energy emotions  
- **Fear**: Negative, anticipatory emotions
- **Sadness**: Negative, low-energy emotions
- **Disgust**: Negative, rejection-based emotions
- **Pride**: Positive, achievement-based emotions
- **Guilt**: Negative, self-critical emotions
- **Love**: Positive, connection-based emotions

## 🛠️ Troubleshooting

### Common Issues

**"No module named 'dotenv'" Error**:
```bash
pip install python-dotenv
```

**"API key not found" Error**:
- Ensure `.env` file exists in project root
- Check that API keys are properly set (no quotes needed)
- Verify the API key is valid and has sufficient credits

**"Connection Error"**:
- Check your internet connection
- Verify API endpoint URLs in `.env` file
- Ensure API service is operational

**For Academic Users**:
- Many universities provide research credits for cloud APIs
- Contact your IT department for institutional API access
- Consider using free tiers for initial testing and development

## 🔬 Research Applications

### For Students

- **Thesis Research**: Use CHS as a foundation for emotion recognition studies
- **Course Projects**: Implement emotion analysis in HCI or psychology courses
- **Data Analysis**: Analyze emotional patterns in text corpora
- **Algorithm Development**: Extend the framework for specific research questions

### For Professors

- **Teaching Tool**: Demonstrate geometric approaches to emotion modeling
- **Research Collaboration**: Integrate CHS into existing emotion research
- **Publication Support**: Use results for academic papers and conferences
- **Student Supervision**: Guide students in emotion computing research

### For Researchers

- **Baseline Comparison**: Compare new emotion models against CHS framework
- **Feature Engineering**: Extract geometric emotion features for ML models
- **Cross-validation**: Validate emotion theories using computational methods
- **Interdisciplinary Work**: Bridge computer science and psychology research

## 🛠️ Customization

### Extending the Framework

```python
# Example: Adding custom emotion categories
class ExtendedCoordinateHeartSystem(CoordinateHeartSystem):
    def __init__(self):
        super().__init__()
        # Add custom initialization
    
    def analyze_custom_emotions(self, text):
        # Implement custom analysis logic
        pass
```

### Configuration Options

- Modify emotion weights in the coordinate mapping
- Adjust stability calculation parameters
- Customize contextual drain factors
- Extend output formatting options

## 📈 Performance Considerations

- **LLM Response Time**: Depends on model complexity and server load
- **CHS Analysis Time**: Typically < 1ms for standard text inputs
- **Memory Usage**: Minimal for single analysis, scalable for batch processing
- **Accuracy**: Validated against psychological emotion models (see research paper)

## 🤝 Contributing

We welcome contributions from the academic community:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow academic coding standards
- Include comprehensive documentation
- Add unit tests for new features
- Cite relevant research in code comments
- Maintain compatibility with existing API

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support & Contact

For academic inquiries, research collaboration, or technical support:

- **Author**: Omar Aldesi
- **Email**: [2441394@std.hu.edu.jo]
- **Research Profile**: [https://scholar.google.com/citations?user=-vS_-XcAAAAJ&hl=en]
- **Issues**: Use GitHub Issues for bug reports and feature requests

## 🙏 Acknowledgments

- Research institutions and colleagues who provided feedback
- Open-source community for foundational tools and libraries
- Students and researchers who tested and validated the framework

## 📚 References

1. Aldesi, O. (2025). "Coordinate Heart System: A Geometric Framework for Emotion Representation." [arXiv preprint arXiv:2507.14593](https://arxiv.org/abs/2507.14593)

---

**For Academic Use**: Please cite the original research paper when using this implementation in your research or publications.

**Version**: 1.0.0  
**Last Updated**: July 2025
