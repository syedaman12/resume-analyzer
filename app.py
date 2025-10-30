from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
import re
import PyPDF2
import docx
from datetime import datetime
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import faiss
import pickle
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import uuid
from typing import List, Dict, Any
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CACHE_FOLDER'] = 'cache'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['CACHE_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Global variables for caching
faiss_index = None
resume_embeddings = {}
job_roles_classifier = None
vectorizer = None
nlp = None

# Initialize models
def initialize_models():
    global nlp, job_roles_classifier, vectorizer
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model loaded successfully")
    except OSError:
        print("❌ spaCy model not found. Please install with: python -m spacy download en_core_web_sm")
        nlp = None
    
    global faiss_index
    faiss_index = faiss.IndexFlatIP(300)
    initialize_role_classifier()

def initialize_role_classifier():
    global job_roles_classifier, vectorizer
    try:
        with open('models/role_classifier.pkl', 'rb') as f:
            job_roles_classifier = pickle.load(f)
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Role classifier loaded successfully")
    except:
        print("⚠️ No pre-trained classifier found. Using keyword-based approach.")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        job_roles_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

JOB_ROLES = [
    'Software Engineer', 'Data Scientist', 'Frontend Developer', 'Backend Developer',
    'Full Stack Developer', 'DevOps Engineer', 'Data Analyst', 'Product Manager',
    'UX/UI Designer', 'QA Engineer', 'System Administrator', 'Network Engineer',
    'Security Engineer', 'Mobile Developer', 'Machine Learning Engineer'
]

ROLE_KEYWORDS = {
    'Software Engineer': ['software', 'engineering', 'development', 'programming', 'code', 'java', 'python', 'c++', 'algorithm', 'data structure'],
    'Data Scientist': ['data science', 'machine learning', 'ai', 'statistics', 'analytics', 'python', 'r', 'sql', 'pandas', 'numpy'],
    'Frontend Developer': ['frontend', 'javascript', 'react', 'angular', 'vue', 'css', 'html', 'typescript', 'responsive', 'web development'],
    'Backend Developer': ['backend', 'server', 'api', 'database', 'python', 'java', 'node.js', 'spring', 'rest', 'microservices'],
    'Full Stack Developer': ['full stack', 'mern', 'mean', 'both frontend backend', 'react', 'node', 'mongodb', 'express', 'database'],
    'DevOps Engineer': ['devops', 'ci/cd', 'docker', 'kubernetes', 'aws', 'azure', 'jenkins', 'terraform', 'infrastructure', 'deployment'],
    'Data Analyst': ['data analysis', 'sql', 'excel', 'tableau', 'power bi', 'reporting', 'analytics', 'visualization', 'statistics'],
    'Product Manager': ['product management', 'agile', 'scrum', 'roadmap', 'stakeholder', 'strategy', 'user story', 'backlog', 'prioritization'],
    'UX/UI Designer': ['ux', 'ui', 'design', 'figma', 'sketch', 'user experience', 'wireframe', 'prototype', 'user research', 'usability'],
    'QA Engineer': ['quality assurance', 'testing', 'automated tests', 'selenium', 'qa', 'test cases', 'bug', 'defect', 'test automation'],
    'System Administrator': ['system admin', 'linux', 'windows server', 'administration', 'networking', 'troubleshooting', 'maintenance'],
    'Network Engineer': ['network', 'cisco', 'routing', 'switching', 'security', 'firewall', 'vpn', 'lan', 'wan'],
    'Security Engineer': ['cybersecurity', 'security', 'penetration testing', 'firewall', 'encryption', 'vulnerability', 'threat', 'compliance'],
    'Mobile Developer': ['mobile', 'ios', 'android', 'swift', 'kotlin', 'react native', 'flutter', 'mobile app', 'sdk'],
    'Machine Learning Engineer': ['ml', 'deep learning', 'tensorflow', 'pytorch', 'neural networks', 'ai', 'model training', 'nlp']
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        print(f"✅ Extracted {len(text)} characters from PDF")
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
    return text

def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
        print(f"✅ Extracted {len(text)} characters from DOCX")
    except Exception as e:
        print(f"❌ Error reading DOCX: {e}")
    return text

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
        print(f"✅ Extracted {len(text)} characters from TXT")
        return text
    except Exception as e:
        print(f"❌ Error reading TXT: {e}")
        return ""

def extract_contact_info(text):
    """Extract contact information from resume text"""
    contact = {
        'email': None,
        'phone': None,
        'linkedin': None,
        'github': None,
        'location': None
    }
    
    # Email extraction
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        contact['email'] = emails[0]
    
    # Phone extraction (improved pattern)
    phone_pattern = r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'
    phones = re.findall(phone_pattern, text)
    if phones:
        contact['phone'] = phones[0]
    
    # LinkedIn extraction
    linkedin_pattern = r'(?:linkedin\.com/in/|linkedin\.com/company/)[\w\-]+'
    linkedin_matches = re.findall(linkedin_pattern, text.lower())
    if linkedin_matches:
        contact['linkedin'] = f"https://www.{linkedin_matches[0]}"
    
    # GitHub extraction
    github_pattern = r'github\.com/[\w\-]+'
    github_matches = re.findall(github_pattern, text.lower())
    if github_matches:
        contact['github'] = f"https://www.{github_matches[0]}"
    
    # Location extraction (basic)
    location_pattern = r'\b(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*),\s*[A-Z]{2}\b'
    locations = re.findall(location_pattern, text)
    if locations:
        contact['location'] = locations[0]
    
    print(f"✅ Extracted contact info: {contact}")
    return contact

def extract_skills(text):
    """Extract skills from resume text"""
    common_skills = [
        'python', 'java', 'javascript', 'typescript', 'html', 'css', 'react', 'angular', 
        'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'sql', 'mysql', 
        'postgresql', 'mongodb', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 
        'github', 'jenkins', 'ci/cd', 'machine learning', 'ai', 'deep learning',
        'data analysis', 'tableau', 'power bi', 'excel', 'rest api', 'graphql', 
        'microservices', 'agile', 'scrum', 'devops', 'linux', 'windows', 'bash',
        'shell scripting', 'networking', 'security', 'testing', 'selenium', 'junit',
        'tdd', 'bdd', 'oop', 'functional programming', 'data structures', 'algorithms',
        'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'matplotlib', 'seaborn',
        'jira', 'confluence', 'slack', 'teams', 'communication', 'leadership', 'teamwork'
    ]
    
    found_skills = []
    text_lower = text.lower()
    
    for skill in common_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.append(skill.title())
    
    # Remove duplicates and return
    unique_skills = list(dict.fromkeys(found_skills))
    print(f"✅ Found {len(unique_skills)} skills: {unique_skills[:10]}...")
    return unique_skills[:20]  # Return top 20 skills

def calculate_ats_score(text, job_description=None):
    """Calculate ATS compatibility score"""
    score = 50  # Base score
    
    # Check for important sections
    sections = ['experience', 'education', 'skills', 'projects', 'summary', 'objective']
    section_count = 0
    for section in sections:
        if section in text.lower():
            section_count += 1
            score += 3
    
    # Check length (optimal resume length)
    word_count = len(text.split())
    if 400 <= word_count <= 800:
        score += 15
    elif 300 <= word_count < 400:
        score += 10
    elif 800 < word_count <= 1200:
        score += 5
    elif word_count > 1200:
        score -= 5
    
    # Check for contact info
    contact = extract_contact_info(text)
    if contact['email']:
        score += 5
    if contact['phone']:
        score += 5
    if contact['linkedin']:
        score += 3
    
    # Skills density
    skills = extract_skills(text)
    if len(skills) >= 10:
        score += 10
    elif len(skills) >= 5:
        score += 5
    elif len(skills) >= 3:
        score += 2
    
    # Check for quantifiable achievements
    achievement_indicators = ['increased', 'decreased', 'improved', 'reduced', 'managed', 'led', 'achieved']
    achievement_count = sum(1 for indicator in achievement_indicators if indicator in text.lower())
    score += min(achievement_count * 2, 10)
    
    # Job description match (if provided)
    if job_description:
        jd_skills = extract_skills(job_description)
        matched_skills = set(skills) & set([s.lower() for s in jd_skills])
        match_percentage = len(matched_skills) / max(len(jd_skills), 1) * 100
        score += min(match_percentage / 5, 10)  # Add up to 10 points for JD match
    
    # Ensure score is within bounds
    final_score = min(max(score, 0), 100)
    print(f"✅ Calculated ATS score: {final_score}")
    return final_score

def analyze_role_fit(text, target_role=None):
    """Analyze how well the resume fits different roles"""
    role_scores = {}
    text_lower = text.lower()
    
    if target_role and target_role in ROLE_KEYWORDS:
        # Focus on target role if specified
        roles_to_check = [target_role]
    else:
        # Check all roles if no target specified
        roles_to_check = JOB_ROLES
    
    for role in roles_to_check:
        keywords = ROLE_KEYWORDS.get(role, [])
        matched_keywords = []
        
        for keyword in keywords:
            # Use word boundaries for better matching
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                matched_keywords.append(keyword)
        
        match_percentage = (len(matched_keywords) / max(len(keywords), 1)) * 100
        role_scores[role] = {
            'match_percentage': min(match_percentage, 100),
            'matched_keywords': matched_keywords,
            'total_keywords': len(keywords),
            'matched_count': len(matched_keywords)
        }
    
    # Find best fitting role
    if role_scores:
        best_role = max(role_scores.items(), key=lambda x: x[1]['match_percentage'])
        primary_role = best_role[0]
        fit_score = best_role[1]['match_percentage']
    else:
        primary_role = "General"
        fit_score = 0
    
    print(f"✅ Role fit analysis - Best role: {primary_role} ({fit_score:.1f}%)")
    
    return {
        'primary_role': primary_role,
        'fit_score': fit_score,
        'role_scores': role_scores
    }

def generate_feedback(text, ats_score, role_fit):
    """Generate actionable feedback for the resume"""
    feedback = []
    
    # ATS Score feedback
    if ats_score >= 80:
        feedback.append("✅ Excellent ATS compatibility score! Your resume is well-optimized.")
    elif ats_score >= 70:
        feedback.append("✅ Good ATS score. Your resume should pass through most systems.")
    elif ats_score >= 60:
        feedback.append("⚠️ Average ATS score. Consider optimizing your resume structure.")
    elif ats_score >= 40:
        feedback.append("⚠️ Below average ATS score. Significant improvements needed.")
    else:
        feedback.append("❌ Low ATS score. Major restructuring required.")
    
    # Length feedback
    word_count = len(text.split())
    if word_count < 300:
        feedback.append("❌ Resume is too short - add more details about experience and skills")
    elif word_count < 400:
        feedback.append("⚠️ Resume is somewhat short - consider adding more achievements")
    elif word_count > 1000:
        feedback.append("⚠️ Resume might be too long - consider condensing to key points")
    elif word_count > 800:
        feedback.append("✅ Good resume length - comprehensive but concise")
    else:
        feedback.append("✅ Ideal resume length - detailed but focused")
    
    # Contact info feedback
    contact = extract_contact_info(text)
    if not contact['email']:
        feedback.append("❌ Missing email address - essential for contact")
    else:
        feedback.append("✅ Email address found")
    
    if not contact['phone']:
        feedback.append("⚠️ Consider adding phone number for better accessibility")
    else:
        feedback.append("✅ Phone number found")
    
    if not contact['linkedin']:
        feedback.append("⚠️ Consider adding LinkedIn profile - important for professional networking")
    else:
        feedback.append("✅ LinkedIn profile found")
    
    # Skills feedback
    skills = extract_skills(text)
    if len(skills) < 5:
        feedback.append("❌ Very few skills listed - add more technical and soft skills")
    elif len(skills) < 10:
        feedback.append("⚠️ Limited skills listed - consider adding more relevant skills")
    elif len(skills) >= 15:
        feedback.append("✅ Excellent skills section - comprehensive and relevant")
    else:
        feedback.append("✅ Good number of skills listed")
    
    # Role fit feedback
    if role_fit['fit_score'] >= 80:
        feedback.append(f"✅ Strong fit for {role_fit['primary_role']} role - excellent keyword alignment")
    elif role_fit['fit_score'] >= 60:
        feedback.append(f"✅ Good fit for {role_fit['primary_role']} role - solid keyword matching")
    elif role_fit['fit_score'] >= 40:
        feedback.append(f"⚠️ Moderate fit for {role_fit['primary_role']} - consider adding more role-specific keywords")
    else:
        feedback.append(f"❌ Weak fit for target roles - add more relevant keywords and skills")
    
    # Section completeness
    sections = ['experience', 'education', 'skills']
    missing_sections = [section for section in sections if section not in text.lower()]
    if missing_sections:
        feedback.append(f"❌ Missing important sections: {', '.join(missing_sections)}")
    else:
        feedback.append("✅ All key resume sections present")
    
    # Quantifiable achievements
    achievement_words = ['increased', 'decreased', 'improved', 'reduced', 'achieved', 'managed', 'led']
    has_achievements = any(word in text.lower() for word in achievement_words)
    if not has_achievements:
        feedback.append("⚠️ Add quantifiable achievements with numbers and metrics")
    else:
        feedback.append("✅ Good use of quantifiable achievements")
    
    print(f"✅ Generated {len(feedback)} feedback items")
    return feedback

def generate_pdf_report(analysis_result):
    """Generate PDF report for the analysis"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1e40af')
        )
        story.append(Paragraph("Resume Analysis Report", title_style))
        
        # Basic Information
        story.append(Paragraph("Basic Information", styles['Heading2']))
        basic_data = [
            ['Filename:', analysis_result.get('filename', 'N/A')],
            ['ATS Score:', f"{analysis_result.get('ats_score', 0)}/100"],
            ['Recommended Role:', analysis_result.get('role_fit', {}).get('primary_role', 'N/A')],
            ['Role Fit Score:', f"{analysis_result.get('role_fit', {}).get('fit_score', 0):.1f}%"]
        ]
        basic_table = Table(basic_data, colWidths=[200, 200])
        basic_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbeafe')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1'))
        ]))
        story.append(basic_table)
        story.append(Spacer(1, 20))
        
        # Contact Information
        story.append(Paragraph("Contact Information", styles['Heading2']))
        contact_info = analysis_result.get('contact', {})
        contact_data = [[key.capitalize(), value or 'Not Found'] for key, value in contact_info.items()]
        contact_table = Table(contact_data, colWidths=[150, 250])
        contact_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbeafe')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1'))
        ]))
        story.append(contact_table)
        story.append(Spacer(1, 20))
        
        # Skills
        story.append(Paragraph("Skills Found", styles['Heading2']))
        skills = analysis_result.get('skills', [])
        if skills:
            skills_text = ", ".join(skills)
            story.append(Paragraph(skills_text, styles['Normal']))
        else:
            story.append(Paragraph("No skills detected", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Feedback
        story.append(Paragraph("Key Feedback", styles['Heading2']))
        feedback_items = analysis_result.get('feedback', [])
        for item in feedback_items:
            # Clean up the feedback item
            clean_item = item.replace('✅', '').replace('⚠️', '').replace('❌', '').strip()
            if '✅' in item:
                story.append(Paragraph(f"• ✓ {clean_item}", styles['Normal']))
            elif '⚠️' in item:
                story.append(Paragraph(f"• ⚠ {clean_item}", styles['Normal']))
            elif '❌' in item:
                story.append(Paragraph(f"• ✗ {clean_item}", styles['Normal']))
            else:
                story.append(Paragraph(f"• {clean_item}", styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        print(f"❌ PDF generation error: {e}")
        return None

# ------------------ Flask Routes ------------------

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400
        
        file = request.files['resume']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            print(f"🔍 Analyzing resume: {filename}")
            
            # Extract text based on file type
            if filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif filename.lower().endswith('.docx'):
                text = extract_text_from_docx(file_path)
            else:
                text = extract_text_from_txt(file_path)
            
            if not text or not text.strip():
                # Clean up file
                try:
                    os.remove(file_path)
                except:
                    pass
                return jsonify({'error': 'Could not extract text from file. The file might be empty, corrupted, or in an unsupported format.'}), 400
            
            # Get optional parameters
            job_description = request.form.get('job_description', '')
            target_role = request.form.get('target_role', '')
            
            print(f"📊 Starting analysis with {len(text)} characters...")
            
            # Perform analysis
            contact_info = extract_contact_info(text)
            skills = extract_skills(text)
            ats_score = calculate_ats_score(text, job_description)
            role_fit = analyze_role_fit(text, target_role)
            feedback = generate_feedback(text, ats_score, role_fit)
            
            # Prepare response
            result = {
                'filename': filename,
                'contact': contact_info,
                'skills': skills,
                'ats_score': ats_score,
                'role_fit': role_fit,
                'feedback': feedback,
                'text_preview': text[:500] + '...' if len(text) > 500 else text,
                'word_count': len(text.split()),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"⚠️ Could not remove file: {e}")
            
            print(f"✅ Analysis completed successfully for {filename}")
            return jsonify(result)
        
        return jsonify({'error': 'Invalid file type. Please upload PDF, DOCX, or TXT files.'}), 400
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        # Clean up file in case of error
        try:
            if 'file_path' in locals():
                os.remove(file_path)
        except:
            pass
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        data = request.get_json()
        if not data or 'analysis_result' not in data:
            return jsonify({'error': 'No analysis result provided'}), 400
        
        analysis_result = data['analysis_result']
        pdf_buffer = generate_pdf_report(analysis_result)
        
        if pdf_buffer:
            return send_file(
                pdf_buffer,
                as_attachment=True,
                download_name=f"resume_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mimetype='application/pdf'
            )
        else:
            return jsonify({'error': 'Failed to generate PDF report'}), 500
            
    except Exception as e:
        print(f"❌ Download report error: {e}")
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': nlp is not None
    })

# Initialize models when app starts
initialize_models()

if __name__ == '__main__':
    print("🚀 Starting Resume Analyzer Pro...")
    print("📁 Upload folder:", app.config['UPLOAD_FOLDER'])
    print("📁 Cache folder:", app.config['CACHE_FOLDER'])
    print("🌐 Server running on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)