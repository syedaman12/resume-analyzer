#!/bin/bash
echo "🚀 Starting build process for Resume Analyzer Pro..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo "🔧 Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads cache models templates

echo "✅ Build completed successfully!"
