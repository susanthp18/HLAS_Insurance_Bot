#!/usr/bin/env python3
"""
Crawling Agent - Enhanced Comprehensive Content Extractor
========================================================

This script combines all extraction functionalities with enhanced FAQ deduplication:
1. Extract FAQs from webpages
2. Extract Benefits and Coverage tables
3. Extract Terms & Conditions PDFs
4. Parse PDFs using LlamaParse and save as markdown
5. Process markdown tables in parsed content
6. Advanced FAQ deduplication across all extracted content

Usage: python crawling_agent.py <url>
"""

import os
import sys

# Add the project root to the Python path before other imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests
import argparse
import logging
from datetime import datetime
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import google.generativeai as genai
from difflib import SequenceMatcher
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION AND LOGGING SETUP
# ============================================================================

class CrawlingAgentConfig:
    """Configuration management for the crawling agent"""

    def __init__(self):
        # API Keys
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.llamaparse_api_key = os.getenv('LLAMAPARSE_API_KEY')

        # Processing Configuration
        self.similarity_threshold = os.getenv('SIMILARITY_THRESHOLD')
        self.api_delay_seconds = os.getenv('API_DELAY_SECONDS')
        self.request_timeout_seconds = os.getenv('REQUEST_TIMEOUT_SECONDS')

        # Logging Configuration
        self.log_level = os.getenv('LOG_LEVEL')
        self.log_file = os.getenv('LOG_FILE')

        # Validate required API keys
        self._validate_config()

    def _validate_config(self):
        """Validate required configuration"""
        missing = []
        required_vars = {
            'GEMINI_API_KEY': self.gemini_api_key,
            'SIMILARITY_THRESHOLD': self.similarity_threshold,
            'API_DELAY_SECONDS': self.api_delay_seconds,
            'REQUEST_TIMEOUT_SECONDS': self.request_timeout_seconds,
            'LOG_LEVEL': self.log_level,
            'LOG_FILE': self.log_file,
        }
        for name, value in required_vars.items():
            if not value:
                missing.append(name)
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        # Optional key
        if not self.llamaparse_api_key:
            logging.warning("LLAMAPARSE_API_KEY not found. PDF parsing will be limited.")

        # Cast after validation
        self.similarity_threshold = float(self.similarity_threshold)
        self.api_delay_seconds = int(self.api_delay_seconds)
        self.request_timeout_seconds = int(self.request_timeout_seconds)

# Initialize configuration
config = CrawlingAgentConfig()

def setup_logging():
    """Setup comprehensive logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(config.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Setup file handler
    file_handler = logging.FileHandler(config.log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.log_level))
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler],
        format=log_format,
        datefmt=date_format
    )

    # Create logger for this module
    logger = logging.getLogger('crawling_agent')
    logger.info("Logging system initialized")
    logger.info(f"Log level: {config.log_level}")
    logger.info(f"Log file: {config.log_file}")

    return logger

# Initialize logging
logger = setup_logging()

# Configure Gemini
try:
    genai.configure(api_key=config.gemini_api_key)
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    raise

# Try to import LlamaParse
try:
    from llama_cloud_services import LlamaParse
    LLAMAPARSE_AVAILABLE = True
    logger.info("LlamaParse library imported successfully")
except ImportError:
    LLAMAPARSE_AVAILABLE = False
    logger.warning("LlamaParse not available. PDF parsing will be skipped.")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_product_name_from_url(url):
    """Extract product name from URL for folder organization"""
    logger.debug(f"Extracting product name from URL: {url}")

    url_patterns = {
        'travel-insurance': 'Travel',
        'maid-insurance': 'Maid',
        'home-protect360': 'Home',
        'family-protect360': 'Family',
        'early-critical-illness': 'Early',
        'hospital-protect360': 'Hospital',
        'car-insurance': 'Car',
        'critical-illness': 'Critical',
        'accident-protect360': 'Accident',
        'fire-insurance': 'Fire',
        'mobile-phone': 'Mobile'
    }

    url_lower = url.lower()
    for pattern, name in url_patterns.items():
        if pattern in url_lower:
            logger.info(f"Product identified: {name} (pattern: {pattern})")
            return name

    # Fallback: extract from URL path
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split('/') if p]
    if path_parts:
        fallback_name = path_parts[-1].replace('-', '_').title()
        logger.warning(f"Using fallback product name: {fallback_name}")
        return fallback_name

    logger.warning("Could not determine product name, using 'Unknown'")
    return "Unknown"

def create_folder_structure(product_name):
    """Create folder structure for all outputs under source_db"""
    logger.info(f"Creating folder structure for product: {product_name}")

    folders = {
        'faqs': f"source_db/FAQ",
        'benefits': f"source_db/benefits",
        'policy': f"source_db/policy",
        'pdfs': f"source_db/pdfs/{product_name}"
    }

    for folder_path in folders.values():
        try:
            os.makedirs(folder_path, exist_ok=True)
            logger.debug(f"Ensured folder exists: {folder_path}")
            print(f"📁 Ensured folder exists: {folder_path}")
        except Exception as e:
            logger.error(f"Failed to create folder {folder_path}: {e}")
            raise

    logger.info("Folder structure created successfully")
    return folders

def get_webpage_content(url):
    """Get webpage content from HTTP(S) or local file using BeautifulSoup.

    If the site renders pop-ups/banners that block content, we fall back to a
    headless browser via Playwright to auto-dismiss common overlays before
    extracting the HTML.
    """
    logger.info(f"Fetching webpage content from: {url}")

    # Local file support: file://C:/path/to/file.html
    try:
        parsed = urlparse(url)
        if parsed.scheme == 'file':
            local_path = parsed.path
            if os.name == 'nt' and local_path.startswith('/') and ':' in local_path[1:3]:
                # file:///C:/... -> /C:/... -> normalize
                local_path = local_path.lstrip('/')
            try:
                with open(local_path, 'rb') as f:
                    content = f.read()
                soup = BeautifulSoup(content, 'html.parser')
                logger.info(f"Loaded local HTML file: {local_path} ({len(content)} bytes)")
                return soup
            except Exception as e:
                error_msg = f"Error reading local file: {str(e)}"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                return None
    except Exception:
        pass

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

        logger.debug(f"Making HTTP request with timeout: {config.request_timeout_seconds}s")
        response = requests.get(url, headers=headers, timeout=config.request_timeout_seconds)
        response.raise_for_status()

        logger.info(f"Successfully fetched content. Status: {response.status_code}, Size: {len(response.content)} bytes")

        soup = BeautifulSoup(response.content, 'html.parser')
        logger.debug("Content parsed with BeautifulSoup")
        # Heuristic: if the DOM is very small or contains typical overlay markers, try browser fallback
        text_len = len(soup.get_text(strip=True)) if soup else 0
        has_overlay_hint = soup.find(class_=re.compile(r"(popup|modal|overlay|banner|gdpr|cookie)", re.I)) if soup else None
        if text_len < 1000 or has_overlay_hint:
            logger.info("Attempting headless browser fetch to bypass pop-ups/overlays")
            html = fetch_with_playwright(url)
            if html:
                return BeautifulSoup(html, 'html.parser')
        return soup

    except requests.exceptions.Timeout:
        error_msg = f"Request timeout after {config.request_timeout_seconds} seconds"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return None
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error fetching content: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        # Try headless browser as a fallback for network-layer blocks that a browser might pass
        try:
            html = fetch_with_playwright(url)
            if html:
                return BeautifulSoup(html, 'html.parser')
        except Exception:
            pass
        return None
    except Exception as e:
        error_msg = f"Unexpected error fetching content: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return None


def fetch_with_playwright(url: str) -> str | None:
    """Use Playwright headless Chromium to load the page, dismiss pop-ups, and return the HTML."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        logger.warning(f"Playwright not available: {e}")
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            page.set_default_timeout(15000)
            page.goto(url, wait_until="networkidle")

            # Common dismissors
            selectors = [
                "button[aria-label='Close']",
                "button[aria-label='Dismiss']",
                "button:has-text('Close')",
                "button:has-text('Got it')",
                "button:has-text('Accept')",
                "button:has-text('I Agree')",
                "#onetrust-accept-btn-handler",
                ".ot-sdk-container .accept-btn",
                ".cookie-accept, .cookie__accept, .btn-accept",
                ".modal .close, .overlay .close, .popup .close",
            ]
            for sel in selectors:
                try:
                    if page.locator(sel).count() > 0:
                        page.locator(sel).first.click()
                except Exception:
                    continue

            # Remove overlays by style if still blocking
            page.evaluate("""
                () => {
                    const killers = ['popup','modal','overlay','banner','cookie','gdpr'];
                    document.querySelectorAll('*').forEach(el => {
                        const c = (el.className || '').toString().toLowerCase();
                        if (killers.some(k => c.includes(k))) {
                            el.style.display = 'none';
                        }
                    });
                }
            """)

            # Wait for content
            page.wait_for_timeout(1000)
            html = page.content()
            context.close()
            browser.close()
            return html
    except Exception as e:
        logger.warning(f"Playwright fetch failed: {e}")
        return None

# ============================================================================
# ENHANCED FAQ DEDUPLICATION FUNCTIONS
# ============================================================================

def calculate_similarity(text1, text2):
    """Calculate similarity between two text strings"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def normalize_question(question):
    """Normalize question text for better comparison"""
    # Remove common question prefixes
    question = re.sub(r'^(Q:|Question:|Q\.|\d+\.)\s*', '', question, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    question = re.sub(r'\s+', ' ', question).strip()
    
    # Convert to lowercase for comparison
    return question.lower()

def normalize_answer(answer):
    """Normalize answer text for better comparison"""
    # Remove common answer prefixes
    answer = re.sub(r'^(A:|Answer:|A\.)\s*', '', answer, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    return answer

def are_questions_similar(q1, q2, threshold=0.85):
    """Check if two questions are similar based on threshold"""
    norm_q1 = normalize_question(q1)
    norm_q2 = normalize_question(q2)
    
    # Calculate similarity
    similarity = calculate_similarity(norm_q1, norm_q2)
    
    return similarity >= threshold

def are_answers_similar(a1, a2, threshold=0.80):
    """Check if two answers are similar based on threshold"""
    norm_a1 = normalize_answer(a1)
    norm_a2 = normalize_answer(a2)
    
    # Calculate similarity
    similarity = calculate_similarity(norm_a1, norm_a2)
    
    return similarity >= threshold

def choose_better_qa_pair(qa1, qa2):
    """Choose the better Q&A pair between two similar ones"""
    # Prefer the one with longer, more detailed answer
    if len(qa2['A']) > len(qa1['A']) * 1.2:  # 20% longer
        return qa2
    elif len(qa1['A']) > len(qa2['A']) * 1.2:
        return qa1
    
    # Prefer the one with more complete question
    if len(qa2['Q']) > len(qa1['Q']) * 1.1:  # 10% longer
        return qa2
    elif len(qa1['Q']) > len(qa2['Q']) * 1.1:
        return qa1
    
    # Default to first one
    return qa1

def advanced_faq_deduplication(faqs_list):
    """Perform advanced deduplication of FAQ pairs"""
    if not faqs_list:
        return []
    
    print(f"🔍 Starting advanced deduplication of {len(faqs_list)} FAQs...")
    
    deduplicated = []
    skipped_count = 0
    
    for current_faq in faqs_list:
        is_duplicate = False
        
        # Check against all existing deduplicated FAQs
        for i, existing_faq in enumerate(deduplicated):
            # Check if questions are similar
            if are_questions_similar(current_faq['Q'], existing_faq['Q']):
                # Questions are similar, check answers
                if are_answers_similar(current_faq['A'], existing_faq['A']):
                    # Both question and answer are similar - this is a duplicate
                    # Choose the better version
                    better_faq = choose_better_qa_pair(existing_faq, current_faq)
                    deduplicated[i] = better_faq
                    is_duplicate = True
                    skipped_count += 1
                    break
                else:
                    # Similar questions but different answers - keep both but note it
                    print(f"   ⚠️ Similar questions with different answers found:")
                    print(f"      Q1: {existing_faq['Q'][:60]}...")
                    print(f"      Q2: {current_faq['Q'][:60]}...")
        
        if not is_duplicate:
            deduplicated.append(current_faq)
    
    print(f"✅ Deduplication complete: {len(deduplicated)} unique FAQs (removed {skipped_count} duplicates)")
    return deduplicated

def read_existing_faqs(filename):
    """Read existing FAQs from file"""
    if not os.path.exists(filename):
        return []
    
    faqs = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Parse Q&A pairs from the file
        lines = content.split('\n')
        current_question = ""
        current_answer = ""
        in_answer = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Q:'):
                # Save previous Q&A pair if exists
                if current_question and current_answer:
                    faqs.append({
                        'Q': current_question.strip(),
                        'A': current_answer.strip()
                    })
                
                # Start new question
                current_question = line[2:].strip()
                current_answer = ""
                in_answer = False
                
            elif line.startswith('A:'):
                # Start answer
                current_answer = line[2:].strip()
                in_answer = True
                
            elif in_answer and line:
                # Continue building answer
                current_answer += " " + line
        
        # Don't forget the last Q&A pair
        if current_question and current_answer:
            faqs.append({
                'Q': current_question.strip(),
                'A': current_answer.strip()
            })
        
        print(f"📖 Read {len(faqs)} existing FAQs from {filename}")
        return faqs
        
    except Exception as e:
        print(f"❌ Error reading existing FAQs: {str(e)}")
        return []

# ============================================================================
# FAQ EXTRACTION FUNCTIONS (Enhanced)
# ============================================================================

def find_faq_sections(soup):
    """Find actual FAQ sections on the webpage"""
    faq_sections = []

    if not soup:
        return faq_sections

    # Common FAQ section indicators
    faq_indicators = [
        'faq', 'frequently asked questions', 'questions and answers', 'q&a', 'qa',
        'common questions', 'help', 'support'
    ]

    # Method 1: Look for elements with FAQ-related classes or IDs
    for indicator in faq_indicators:
        elements = soup.find_all(class_=re.compile(indicator, re.IGNORECASE))
        faq_sections.extend(elements)
        elements = soup.find_all(id=re.compile(indicator, re.IGNORECASE))
        faq_sections.extend(elements)

    # Method 2: Look for headings that contain FAQ-related text
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    for heading in headings:
        heading_text = heading.get_text().lower()
        if any(indicator in heading_text for indicator in faq_indicators):
            parent = heading.parent
            if parent:
                faq_sections.append(parent)

    # Method 3: Look for accordion/collapse structures
    accordion_classes = ['accordion', 'collapse', 'toggle', 'expandable', 'dropdown']
    for acc_class in accordion_classes:
        elements = soup.find_all(class_=re.compile(acc_class, re.IGNORECASE))
        faq_sections.extend(elements)

    # Method 4: Look for question patterns in the text
    all_elements = soup.find_all(['div', 'section', 'article'])
    for element in all_elements:
        text = element.get_text()
        question_count = text.count('?')
        if question_count >= 3:  # Likely an FAQ section
            faq_sections.append(element)

    return faq_sections

def extract_qa_pairs(element):
    """Extract Q&A pairs from an FAQ element"""
    qa_pairs = []

    if not element:
        return qa_pairs

    # Method 1: Look for explicit Q: and A: patterns
    text = element.get_text()
    lines = text.split('\n')

    current_question = ""
    current_answer = ""
    in_answer = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for question patterns
        if (line.startswith('Q:') or line.startswith('Question:') or
            line.startswith('Q.') or re.match(r'^\d+\.', line) and '?' in line):

            # Save previous Q&A pair if exists
            if current_question and current_answer:
                qa_pairs.append({
                    'Q': current_question.strip(),
                    'A': current_answer.strip()
                })

            # Start new question
            current_question = re.sub(r'^(Q:|Question:|Q\.|\d+\.)\s*', '', line)
            current_answer = ""
            in_answer = False

        elif (line.startswith('A:') or line.startswith('Answer:') or
              line.startswith('A.')):

            # Start answer
            current_answer = re.sub(r'^(A:|Answer:|A\.)\s*', '', line)
            in_answer = True

        elif '?' in line and len(line) > 10 and not in_answer:
            # Potential question without explicit Q: marker
            if current_question and current_answer:
                qa_pairs.append({
                    'Q': current_question.strip(),
                    'A': current_answer.strip()
                })

            current_question = line
            current_answer = ""
            in_answer = False

        elif in_answer and line:
            # Continue building answer
            current_answer += " " + line

        elif current_question and not in_answer and line and not '?' in line:
            # This might be the answer without explicit A: marker
            current_answer += " " + line
            in_answer = True

    # Don't forget the last Q&A pair
    if current_question and current_answer:
        qa_pairs.append({
            'Q': current_question.strip(),
            'A': current_answer.strip()
        })

    # Clean up and basic deduplication
    cleaned_pairs = []
    seen_questions = set()

    for pair in qa_pairs:
        question = re.sub(r'\s+', ' ', pair['Q']).strip()
        answer = re.sub(r'\s+', ' ', pair['A']).strip()

        if (len(question) > 10 and len(answer) > 20 and
            question.lower() not in seen_questions):

            cleaned_pairs.append({
                'Q': question,
                'A': answer
            })
            seen_questions.add(question.lower())

    return cleaned_pairs

def extract_faqs_enhanced(soup, product_name):
    """Enhanced FAQ extraction with advanced deduplication"""
    logger.info(f"Starting enhanced FAQ extraction for product: {product_name}")
    print("🔍 Extracting FAQs with enhanced deduplication...")

    # Find potential FAQ sections
    faq_sections = find_faq_sections(soup)
    logger.info(f"Found {len(faq_sections)} potential FAQ sections")
    print(f"🔎 Found {len(faq_sections)} potential FAQ sections")

    all_faqs = []

    # Extract FAQs from each section
    for i, section in enumerate(faq_sections):
        qa_pairs = extract_qa_pairs(section)
        all_faqs.extend(qa_pairs)

        if qa_pairs:
            print(f"   ✅ Found {len(qa_pairs)} Q&A pairs in section {i+1}")

    # If no structured FAQs found, try extracting from the entire page
    if not all_faqs:
        print("🔍 No structured FAQs found, scanning entire page...")
        body = soup.find('body')
        if body:
            qa_pairs = extract_qa_pairs(body)
            all_faqs.extend(qa_pairs)

    # Read existing FAQs if file exists
    filename = f"source_db/FAQ/{product_name}_FAQs.txt"
    existing_faqs = read_existing_faqs(filename)

    # Combine new and existing FAQs
    combined_faqs = existing_faqs + all_faqs
    print(f"📊 Combined total: {len(combined_faqs)} FAQs (existing: {len(existing_faqs)}, new: {len(all_faqs)})")

    # Perform advanced deduplication
    unique_faqs = advanced_faq_deduplication(combined_faqs)

    # Save enhanced FAQs
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            if not unique_faqs:
                file.write("No FAQs found on this page.\n")
            else:
                for faq in unique_faqs:
                    file.write(f"Q: {faq['Q']}\n")
                    file.write(f"A: {faq['A']}\n\n")

        print(f"💾 Saved {len(unique_faqs)} deduplicated FAQs to: {filename}")
        return len(unique_faqs)

    except Exception as e:
        print(f"❌ Error saving FAQs: {str(e)}")
        return 0

# ============================================================================
# TABLE EXTRACTION FUNCTIONS
# ============================================================================

def extract_tables_from_page(soup):
    """Extract all tables from the webpage with support for complex structures"""
    if not soup:
        return []

    tables = []
    table_elements = soup.find_all('table')

    for i, table in enumerate(table_elements):
        rows = table.find_all('tr')
        if len(rows) < 2:  # Need at least header + 1 data row
            continue

        table_data = {
            'table_number': i + 1,
            'html': str(table),
            'structured_data': extract_complex_table_structure(table),
            'raw_rows': []
        }

        # Extract raw rows as fallback
        for row in rows:
            row_data = []
            for cell in row.find_all(['td', 'th']):
                cell_text = cell.get_text(strip=True)
                colspan = int(cell.get('colspan', 1))
                rowspan = int(cell.get('rowspan', 1))
                row_data.append({
                    'text': cell_text,
                    'colspan': colspan,
                    'rowspan': rowspan,
                    'is_header': cell.name == 'th'
                })
            if row_data:
                table_data['raw_rows'].append(row_data)

        if len(table_data['raw_rows']) > 1:
            tables.append(table_data)

    return tables

def extract_complex_table_structure(table):
    """Extract complex table structure with hierarchical headers"""
    structure = {
        'headers': [],
        'sub_headers': [],
        'data_rows': [],
        'column_mapping': {}
    }

    rows = table.find_all('tr')
    if not rows:
        return structure

    # Identify header rows
    header_rows = []
    data_start_idx = 0

    for i, row in enumerate(rows):
        cells = row.find_all(['th', 'td'])
        has_headers = any(cell.name == 'th' for cell in cells)
        if has_headers or i < 2:
            header_rows.append(row)
            data_start_idx = i + 1
        else:
            break

    # Process header rows
    if header_rows:
        # First header row (main headers)
        main_headers = []
        first_row_cells = header_rows[0].find_all(['th', 'td'])
        for cell in first_row_cells:
            text = cell.get_text(strip=True)
            colspan = int(cell.get('colspan', 1))
            main_headers.append({
                'text': text,
                'colspan': colspan,
                'start_col': len([h for h in main_headers if h]),
                'end_col': len([h for h in main_headers if h]) + colspan - 1
            })
        structure['headers'] = main_headers

        # Second header row (sub-headers) if exists
        if len(header_rows) > 1:
            sub_headers = []
            second_row_cells = header_rows[1].find_all(['th', 'td'])
            for cell in second_row_cells:
                text = cell.get_text(strip=True)
                sub_headers.append(text)
            structure['sub_headers'] = sub_headers

    # Process data rows
    for row in rows[data_start_idx:]:
        cells = row.find_all(['td', 'th'])
        row_data = []
        for cell in cells:
            text = cell.get_text(strip=True)
            if text:
                row_data.append(text)
        if row_data:
            structure['data_rows'].append(row_data)

    return structure

def create_table_processing_prompt(table_data, product_name):
    """Create a comprehensive prompt for processing complex table data"""
    structured = table_data['structured_data']
    raw_rows = table_data['raw_rows']

    prompt = f"""
You are an expert at converting complex insurance table data into clear, standalone sentences for a retrieval system.

TASK: Convert the following {product_name} insurance table into self-contained sentences.

TABLE STRUCTURE:
"""

    # Add main headers information
    if structured['headers']:
        prompt += "MAIN HEADERS: "
        for header in structured['headers']:
            if header['colspan'] > 1:
                prompt += f"'{header['text']}' (spans {header['colspan']} columns), "
            else:
                prompt += f"'{header['text']}', "
        prompt += "\n"

    # Add sub-headers information
    if structured['sub_headers']:
        prompt += f"SUB-HEADERS: {structured['sub_headers']}\n"

    # Add raw table data for context
    prompt += "\nRAW TABLE DATA:\n"
    for i, row in enumerate(raw_rows):
        row_texts = []
        for cell in row:
            if cell['text']:
                cell_info = cell['text']
                if cell['colspan'] > 1:
                    cell_info += f" [spans {cell['colspan']} cols]"
                if cell['rowspan'] > 1:
                    cell_info += f" [spans {cell['rowspan']} rows]"
                if cell['is_header']:
                    cell_info += " [HEADER]"
                row_texts.append(cell_info)
        if row_texts:
            prompt += f"Row {i+1}: {' | '.join(row_texts)}\n"

    prompt += f"""

CRITICAL INSTRUCTIONS:
1. This table has HIERARCHICAL STRUCTURE with main sections and sub-categories
2. Include ALL PLAN LEVELS (Basic, Enhanced, Premier, Exclusive) in your sentences when coverage amounts differ
3. Preserve FREQUENCY/CONDITION information (per visit, per accident, per year)
4. Handle HIERARCHICAL SECTIONS properly:
   - Main categories (e.g., "Personal Accident")
   - Sub-categories (e.g., "A. Accidental Death", "B. Permanent Disablement")
   - Sub-sub-categories (e.g., "Clinical Visit", "Dental", "Ambulance Fee")
5. Expand ALL abbreviations and codes into full terminology
6. Make each sentence completely self-contained with full context
7. Include specific dollar amounts for EACH plan level when they differ
8. Mention the specific conditions (per visit, per accident, per year) in each sentence

EXAMPLE FORMAT FOR HIERARCHICAL DATA:
Instead of: "Clinical Visit | per visit | 50 | 75 | 100 | 200"
Write: "Under the Accident Medical Reimbursement coverage of {product_name} insurance, Clinical Visit benefits are provided per visit with coverage of $50 for Basic plan, $75 for Enhanced plan, $100 for Premier plan, and $200 for Exclusive plan."

OUTPUT: Provide only the converted sentences, one per line, without any table formatting or additional commentary.
"""

    return prompt

def process_with_azure_openai(prompt):
    """Process a prompt using Azure OpenAI GPT-4 as a fallback."""
    logger.info("Falling back to Azure OpenAI GPT-4.")
    try:
        azure_llm = AzureChatOpenAI(
            temperature=0,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment="gpt-4",  # Explicitly use gpt-4 as requested
        )
        response = azure_llm.invoke(prompt)
        logger.info("Successfully processed with Azure OpenAI GPT-4.")
        return response.content
    except Exception as e:
        logger.error(f"Azure OpenAI (GPT-4) fallback also failed: {e}")
        return None

def process_table_with_gemini(table_data, product_name):
    """Process table data using Gemini 2.5-Pro with Azure fallback"""
    logger.debug(f"Processing table with Gemini for product: {product_name}")
    prompt = create_table_processing_prompt(table_data, product_name)
    logger.debug(f"Generated prompt length: {len(prompt)} characters")

    response_text = None
    try:
        logger.info("Attempting to process with Gemini...")
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        logger.debug("Received response from Gemini API")
        response_text = response.text
    except Exception as e:
        logger.warning(f"Gemini processing failed: {e}. Falling back to Azure OpenAI.")
        print("⚠️ Gemini failed, falling back to Azure OpenAI GPT-4...")
        response_text = process_with_azure_openai(prompt)

    if response_text:
        sentences = []
        lines = response_text.strip().split('\n')
        logger.debug(f"Processing {len(lines)} lines from LLM response")

        for line in lines:
            line = line.strip()
            if line and not line.startswith('*') and not line.startswith('-') and len(line) > 20:
                line = re.sub(r'^\d+\.\s*', '', line)
                line = re.sub(r'^•\s*', '', line)
                line = re.sub(r'^\-\s*', '', line)
                sentences.append(line)

        logger.info(f"Generated {len(sentences)} sentences from LLM response")
        return sentences
    else:
        logger.error("Both Gemini and Azure OpenAI failed to generate a response.")
        print("❌ Both Gemini and Azure OpenAI failed.")
        return []

def extract_tables(soup, product_name):
    """Extract and process all tables from the webpage"""
    print("📊 Extracting tables...")

    tables = extract_tables_from_page(soup)
    print(f"🔎 Found {len(tables)} tables on the page")

    all_table_sentences = []

    # Process each table with Gemini
    for i, table_data in enumerate(tables):
        try:
            print(f"🤖 Processing table {i+1}/{len(tables)} with Gemini 2.5-Pro...")

            # Add table context
            table_info = f"TABLE {i+1} FROM {product_name.upper()} INSURANCE:"
            all_table_sentences.append(table_info)

            # Process with Gemini
            sentences = process_table_with_gemini(table_data, product_name)

            if sentences:
                all_table_sentences.extend(sentences)
                print(f"   ✅ Generated {len(sentences)} sentences from table {i+1}")
            else:
                print(f"   ⚠️ No sentences generated from table {i+1}")

            all_table_sentences.append("")  # Empty line between tables
            logger.debug(f"Applying API delay: {config.api_delay_seconds}s")
            time.sleep(config.api_delay_seconds)  # Respect API limits

        except KeyboardInterrupt:
            print(f"\n⚠️ Processing interrupted at table {i+1}")
            break
        except Exception as e:
            print(f"   ❌ Error processing table {i+1}: {str(e)}")
            continue

    # Save tables to benefits folder
    filename = f"source_db/benefits/{product_name}_benefits.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            if not all_table_sentences or len([s for s in all_table_sentences if s.strip()]) == 0:
                file.write("No tables found on this page.\n")
            else:
                for sentence in all_table_sentences:
                    if sentence.strip():
                        file.write(f"{sentence}\n")
                    else:
                        file.write("\n")

        content_count = len([s for s in all_table_sentences if s and not s.startswith('TABLE') and s.strip()])
        print(f"💾 Saved {content_count} table sentences to: {filename}")
        return content_count

    except Exception as e:
        print(f"❌ Error saving tables: {str(e)}")
        return 0

# ============================================================================
# PDF EXTRACTION FUNCTIONS
# ============================================================================

def find_pdf_links(soup, base_url):
    """Find all PDF links on the webpage, focusing on terms and conditions"""
    pdf_links = []

    if not soup:
        return pdf_links

    # Terms and conditions keywords
    tc_keywords = [
        'terms and conditions', 'terms & conditions', 'tnc', 't&c', 'tc',
        'policy terms', 'policy conditions', 'policy document', 'policy wording',
        'product disclosure', 'policy schedule', 'coverage details',
        'insurance terms', 'contract terms', 'policy booklet'
    ]

    # Find all links
    all_links = soup.find_all('a', href=True)

    for link in all_links:
        href = link.get('href', '').strip()
        link_text = link.get_text().strip().lower()

        if not href or href.startswith('#') or href.startswith('javascript:'):
            continue

        full_url = urljoin(base_url, href)

        # Check if it's a PDF link
        is_pdf = (href.lower().endswith('.pdf') or
                 'pdf' in link_text or
                 '.pdf' in href.lower())

        if is_pdf:
            # Check if it's related to terms and conditions
            is_tc_related = False

            # Check link text for T&C keywords
            for keyword in tc_keywords:
                if keyword in link_text:
                    is_tc_related = True
                    break

            # Check URL for T&C keywords
            if not is_tc_related:
                url_lower = href.lower()
                for keyword in tc_keywords:
                    keyword_clean = keyword.replace(' ', '').replace('&', '')
                    if keyword_clean in url_lower or keyword.replace(' ', '-') in url_lower:
                        is_tc_related = True
                        break

            # Check parent elements for T&C context
            if not is_tc_related:
                parent = link.parent
                for _ in range(3):  # Check up to 3 parent levels
                    if parent:
                        parent_text = parent.get_text().lower()
                        for keyword in tc_keywords:
                            if keyword in parent_text:
                                is_tc_related = True
                                break
                        if is_tc_related:
                            break
                        parent = parent.parent
                    else:
                        break

            pdf_info = {
                'url': full_url,
                'text': link.get_text().strip(),
                'is_tc_related': is_tc_related,
                'original_href': href
            }
            pdf_links.append(pdf_info)

    return pdf_links

def download_pdf(pdf_url, save_path, filename):
    """Download a PDF file from the given URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*',
        }

        print(f"📥 Downloading: {filename}")
        response = requests.get(pdf_url, headers=headers, timeout=60, stream=True)
        response.raise_for_status()

        # Ensure filename ends with .pdf
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'

        full_path = os.path.join(save_path, filename)

        # Download the file
        with open(full_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        file_size = os.path.getsize(full_path)
        print(f"✅ Downloaded: {filename} ({file_size:,} bytes)")
        return True

    except Exception as e:
        print(f"❌ Error downloading {filename}: {str(e)}")
        return False

def extract_filename_from_url(url):
    """Extract filename from URL"""
    try:
        from urllib.parse import urlparse, unquote
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        filename = unquote(filename)  # Decode URL encoding

        # Only return if it's a valid PDF filename
        if filename and filename.lower().endswith('.pdf') and len(filename) > 4:
            return filename
    except:
        pass
    return None

def sanitize_filename(filename):
    """Sanitize filename for safe file system storage"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    filename = re.sub(r'\s+', ' ', filename).strip()
    if len(filename) > 100:
        filename = filename[:100]

    return filename

def get_best_filename(pdf_info, fallback_name):
    """Get the best filename for a PDF, prioritizing URL filename over link text"""
    # First try to extract filename from URL
    url_filename = extract_filename_from_url(pdf_info['url'])
    if url_filename:
        return sanitize_filename(url_filename)

    # Fall back to link text if available
    if pdf_info['text']:
        return sanitize_filename(pdf_info['text'])

    # Use fallback name
    return fallback_name

def extract_pdfs(soup, base_url, product_name):
    """Extract and download all PDFs from the webpage"""
    print("📄 Extracting PDFs...")

    pdf_links = find_pdf_links(soup, base_url)
    print(f"🔎 Found {len(pdf_links)} potential PDF links")

    # Filter for T&C related PDFs
    tc_pdfs = [pdf for pdf in pdf_links if pdf['is_tc_related']]
    all_pdfs = pdf_links

    print(f"📋 Terms & Conditions related: {len(tc_pdfs)}")
    print(f"📋 All PDFs found: {len(all_pdfs)}")

    downloaded_files = []
    save_folder = f"source_db/pdfs/{product_name}"

    # Download T&C PDFs first (priority)
    for i, pdf_info in enumerate(tc_pdfs):
        try:
            filename = get_best_filename(pdf_info, f"terms_conditions_{i+1}")

            if download_pdf(pdf_info['url'], save_folder, filename):
                downloaded_files.append({
                    'filename': filename if filename.endswith('.pdf') else filename + '.pdf',
                    'url': pdf_info['url'],
                    'type': 'Terms & Conditions'
                })

            time.sleep(1)  # Small delay between downloads

        except Exception as e:
            print(f"❌ Failed to download PDF {i+1}: {str(e)}")

    # Download other PDFs if no T&C PDFs found
    if not tc_pdfs and all_pdfs:
        print("📥 No specific T&C PDFs found, downloading all PDFs...")
        for i, pdf_info in enumerate(all_pdfs[:5]):  # Limit to 5 PDFs
            try:
                filename = get_best_filename(pdf_info, f"document_{i+1}")

                if download_pdf(pdf_info['url'], save_folder, filename):
                    downloaded_files.append({
                        'filename': filename if filename.endswith('.pdf') else filename + '.pdf',
                        'url': pdf_info['url'],
                        'type': 'General Document'
                    })

                time.sleep(1)

            except Exception as e:
                print(f"❌ Failed to download PDF {i+1}: {str(e)}")

    print(f"📊 Total PDFs downloaded: {len(downloaded_files)}")

    # Save download log
    log_filename = os.path.join(save_folder, "download_log.txt")
    try:
        with open(log_filename, 'w', encoding='utf-8') as file:
            file.write(f"PDF Download Log for {product_name}\n")
            file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write("=" * 50 + "\n\n")

            if not downloaded_files:
                file.write("No PDFs were downloaded.\n")
            else:
                for i, file_info in enumerate(downloaded_files, 1):
                    file.write(f"{i}. {file_info['filename']}\n")
                    file.write(f"   Type: {file_info['type']}\n")
                    file.write(f"   URL: {file_info['url']}\n\n")

        print(f"📝 Download log saved: {log_filename}")
    except Exception as e:
        print(f"❌ Error saving download log: {str(e)}")

    return downloaded_files

# ============================================================================
# LLAMAPARSE PROCESSING FUNCTIONS
# ============================================================================

def setup_llamaparse():
    """Setup LlamaParse with comprehensive prompting"""
    logger.info("Setting up LlamaParse")

    if not LLAMAPARSE_AVAILABLE:
        logger.warning("LlamaParse library not available")
        return None

    if not config.llamaparse_api_key:
        logger.error("LlamaParse API key not configured")
        return None

    try:
        logger.debug("Initializing LlamaParse with configuration")
        parser = LlamaParse(
            api_key=config.llamaparse_api_key,

            parse_mode="parse_page_with_llm",

            system_prompt_append="""You are an expert document parser specialized in converting complex PDFs into RAG-optimized content. Your primary goal is to create semantically rich, standalone content chunks that can be understood independently without reference to other parts of the document.

Key principles:
- Every output unit must be completely self-contained and contextually complete
- Preserve all information while making it accessible for semantic search
- Transform tabular data into narrative descriptions that capture relationships and meaning
- Expand all references and abbreviations using document context""",

            user_prompt="""Parse this PDF document with extreme attention to creating standalone, contextually complete content chunks. Follow these precise instructions:

**TEXT FLOW AND STRUCTURE:**
- Merge multi-column layouts into natural reading order (left-to-right, top-to-bottom)
- Preserve document hierarchy using markdown headers (# H1, ## H2, ### H3, etc.)
- Each section must begin with its full title and context

**TABLE TRANSFORMATION (CRITICAL):**
For every table, convert each data row into a complete standalone paragraph following this format:
"In the [Table Title/Section Context], [Row Label/Category] has the following characteristics: [Column 1 Header] is [Value 1], [Column 2 Header] is [Value 2], [Column 3 Header] is [Value 3]. [Add any contextual meaning or implications from surrounding text]."

Example transformation:
Instead of: "Product A | 15% | $100 | Available"
Write: "In the Product Pricing Table, Product A has the following specifications: the interest rate is 15%, the base cost is $100, and the availability status is Available. This product falls under the standard pricing tier mentioned in Section 2.1."

**REFERENCE RESOLUTION:**
- Expand ALL abbreviations, codes, and cross-references using document context
- Replace "ibid", "as mentioned above", "the aforementioned" with specific content
- Convert "see Table X" to "see the [Table Name] which shows [brief description]"
- Mark unresolvable references as [REFERENCE UNCLEAR: original text]

**COMPREHENSIVE INCLUSION:**
- Include ALL footnotes inline using format: [Footnote: complete footnote text]
- Convert checkboxes to explicit statements: "Option X is selected/checked"
- Include headers, footers, and marginal text if they contain unique information
- Preserve all numerical values in original format (currencies, percentages, units)

**MULTI-PAGE HANDLING:**
- Stitch continuation content across pages seamlessly
- For tables spanning pages, combine into complete entries without duplicate headers
- Maintain context when content continues from previous pages

**SELF-CONTAINED REQUIREMENT:**
Every paragraph, bullet point, and data entry must be independently meaningful. A reader should understand the content without needing to reference other parts of the document. Include necessary context, section names, and background information in each standalone unit.

**QUALITY CHECKS:**
- No orphaned references or incomplete thoughts
- All table data converted to descriptive sentences
- Every abbreviation explained or marked as unclear
- All content chunks include sufficient context for independent comprehension""",

            file_name="here.pdf",
        )

        logger.info("LlamaParse configured successfully")
        return parser

    except Exception as e:
        error_msg = f"Error setting up LlamaParse: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return None

def parse_pdfs_with_llamaparse(product_name, downloaded_files):
    """Parse downloaded PDFs using LlamaParse and save as markdown"""
    if not LLAMAPARSE_AVAILABLE:
        print("⚠️ LlamaParse not available, skipping PDF parsing")
        return 0

    if not downloaded_files:
        print("📄 No PDFs to parse")
        return 0

    print("🤖 Setting up LlamaParse...")
    parser = setup_llamaparse()
    if not parser:
        return 0

    parsed_count = 0
    pdf_folder = f"source_db/pdfs/{product_name}"
    output_folder = f"source_db/policy"

    # Product-specific PDF naming patterns
    pdf_to_parse = None

    # Define product-specific PDF patterns
    product_pdf_patterns = {
        'early': ['Policy Wordings.pdf'],
        'maid': ['Maid-Protect360-Pro_PW_V2_11Jan2024'],  # Specific PDF for Maid product
        'travel': ['Travel_Protect360_PW']
    }

    product_lower = product_name.lower()
    if product_lower in product_pdf_patterns:
        patterns = product_pdf_patterns[product_lower]

        # Look for files matching the product-specific patterns
        if os.path.exists(pdf_folder):
            for filename in os.listdir(pdf_folder):
                if filename.endswith('.pdf'):
                    for pattern in patterns:
                        if filename.startswith(pattern):
                            pdf_to_parse = os.path.join(pdf_folder, filename)
                            print(f"📋 Found product-specific PDF for {product_name}: {filename}")
                            break
                    if pdf_to_parse:
                        break

        if not pdf_to_parse:
            print(f"📋 No product-specific PDF found for {product_name}, will look for here.pdf as fallback")

    # If no specific PDF found yet
    if pdf_to_parse is None:
        # For Travel, do NOT fallback to here.pdf
        if product_lower == 'travel':
            print("📄 Travel: specific PDF Travel_Protect360_PW*.pdf not found. Skipping PDF parsing.")
        else:
            # Look for "here.pdf" file (standard name for parsing)
            here_pdf_path = os.path.join(pdf_folder, "here.pdf")

            # If "here.pdf" doesn't exist, rename the first downloaded PDF
            if not os.path.exists(here_pdf_path) and downloaded_files:
                first_pdf = downloaded_files[0]['filename']
                first_pdf_path = os.path.join(pdf_folder, first_pdf)

                if os.path.exists(first_pdf_path):
                    try:
                        # Copy the first PDF as "here.pdf"
                        import shutil
                        shutil.copy2(first_pdf_path, here_pdf_path)
                        print(f"📋 Copied {first_pdf} as here.pdf for parsing")
                    except Exception as e:
                        print(f"❌ Error copying PDF: {str(e)}")
                        return 0

            if os.path.exists(here_pdf_path):
                pdf_to_parse = here_pdf_path

    if pdf_to_parse:
        try:
            print(f"🤖 Parsing {pdf_to_parse} with LlamaParse...")

            # Parse the PDF
            result = parser.parse(pdf_to_parse)

            # Get markdown documents
            markdown_documents = result.get_markdown_documents(split_by_page=True)

            if markdown_documents:
                # Combine all pages into one markdown file
                combined_content = []

                for i, doc in enumerate(markdown_documents):
                    combined_content.append(f"# Page {i+1}\n\n")
                    combined_content.append(doc.text)
                    combined_content.append("\n\n---\n\n")

                # Save as markdown file
                output_filename = os.path.join(output_folder, f"{product_name}_policy.md")

                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write(''.join(combined_content))

                print(f"✅ Parsed PDF saved as: {output_filename}")
                print(f"📊 Generated {len(markdown_documents)} pages of content")
                parsed_count = 1

            else:
                print("⚠️ No content extracted from PDF")

        except Exception as e:
            print(f"❌ Error parsing PDF with LlamaParse: {str(e)}")

    else:
        print("📄 No PDF file found to parse")

    return parsed_count

# ============================================================================
# MARKDOWN TABLE PROCESSING FUNCTIONS
# ============================================================================

def detect_markdown_tables(content):
    """Detect markdown tables in the content"""
    tables = []
    lines = content.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if this line looks like a table header (contains |)
        if '|' in line and line.count('|') >= 2:
            # Check if the next line is a separator (contains dashes and |)
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if '|' in next_line and ('-' in next_line or '=' in next_line):
                    # This is likely a markdown table
                    table_start = i
                    table_lines = [line, next_line]

                    # Continue reading table rows
                    j = i + 2
                    while j < len(lines):
                        row_line = lines[j].strip()
                        if '|' in row_line and row_line.count('|') >= 2:
                            table_lines.append(row_line)
                            j += 1
                        else:
                            break

                    # Only include tables with at least 3 lines (header, separator, data)
                    if len(table_lines) >= 3:
                        # Get context before the table
                        context_start = max(0, table_start - 5)
                        context_lines = lines[context_start:table_start]
                        context = '\n'.join([l for l in context_lines if l.strip()])

                        tables.append({
                            'start_line': table_start,
                            'end_line': j - 1,
                            'table_lines': table_lines,
                            'context': context,
                            'raw_content': '\n'.join(table_lines)
                        })

                    i = j
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1

    return tables

def parse_markdown_table_structure(table_lines):
    """Parse markdown table structure"""
    if len(table_lines) < 3:
        return None

    # Parse header
    header_line = table_lines[0].strip()
    headers = [cell.strip() for cell in header_line.split('|') if cell.strip()]

    # Parse data rows (skip separator line)
    data_rows = []
    for line in table_lines[2:]:
        if line.strip():
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells:  # Only add non-empty rows
                data_rows.append(cells)

    return {
        'headers': headers,
        'data_rows': data_rows,
        'num_columns': len(headers),
        'num_data_rows': len(data_rows)
    }

def create_markdown_table_prompt(table_data, context, product_name):
    """Create prompt for processing markdown table with Gemini"""

    structure = table_data['structure']

    prompt = f"""
You are an expert at converting markdown table data into clear, standalone sentences for a retrieval system.

TASK: Convert the following {product_name} insurance table into self-contained sentences.

CONTEXT BEFORE TABLE:
{context}

TABLE STRUCTURE:
Headers: {structure['headers']}
Number of columns: {structure['num_columns']}
Number of data rows: {structure['num_data_rows']}

RAW MARKDOWN TABLE:
{table_data['raw_content']}

STRUCTURED DATA:
Headers: {structure['headers']}
"""

    for i, row in enumerate(structure['data_rows']):
        prompt += f"Row {i+1}: {row}\n"

    prompt += f"""

CRITICAL INSTRUCTIONS:
1. Convert each table row into a complete standalone sentence
2. Include the context from the surrounding text to provide full meaning
3. Make each sentence completely self-contained with full context
4. Include specific values and conditions from each row
5. Reference the table context and section it belongs to
6. Expand any abbreviations or codes into full terminology
7. Preserve all numerical values, percentages, and currency amounts exactly

EXAMPLE FORMAT:
Instead of: "Death | 100%"
Write: "Under the {product_name} insurance Covered Events and Benefit Limits, Death coverage provides one hundred percent (100%) of the benefit limit as compensation."

Instead of: "Enhanced | $50"
Write: "For the Enhanced Travel Insurance Plan under {product_name} insurance, the Per Day Limit for COVID-19 isolation orders is fifty dollars ($50) for each continuous twenty-four hour period."

IMPORTANT:
- Every sentence must be independently meaningful
- Include the insurance product name and section context
- Preserve exact monetary amounts and percentages
- Reference the specific conditions or requirements mentioned in the context

OUTPUT: Provide only the converted sentences, one per line, without any table formatting or additional commentary.
"""

    return prompt

def process_markdown_table_with_gemini(table_data, context, product_name):
    """Process markdown table data using Gemini with Azure fallback"""
    prompt = create_markdown_table_prompt(table_data, context, product_name)
    response_text = None
    try:
        logger.info("Attempting to process markdown table with Gemini...")
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        response_text = response.text
    except Exception as e:
        logger.warning(f"Gemini processing for markdown table failed: {e}. Falling back to Azure OpenAI.")
        print("⚠️ Gemini failed for markdown table, falling back to Azure OpenAI GPT-4...")
        response_text = process_with_azure_openai(prompt)

    if response_text:
        sentences = []
        lines = response_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line and not line.startswith('*') and not line.startswith('-') and len(line) > 20:
                line = re.sub(r'^\d+\.\s*', '', line)
                line = re.sub(r'^•\s*', '', line)
                line = re.sub(r'^\-\s*', '', line)
                sentences.append(line)

        return sentences
    else:
        logger.error("Both Gemini and Azure OpenAI failed for markdown table.")
        print("⚠️ No response from Gemini or Azure for markdown table")
        return []

def process_markdown_tables_in_file(file_path, product_name):
    """Process all markdown tables in a file and replace them with standalone sentences"""
    if not os.path.exists(file_path):
        print(f"📄 No markdown file found: {file_path}")
        return 0

    print(f"🔍 Processing markdown tables in: {file_path}")

    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")
        return 0

    # Detect markdown tables
    tables = detect_markdown_tables(content)
    print(f"🔎 Found {len(tables)} markdown tables")

    if not tables:
        return 0

    # Process tables in reverse order to maintain line numbers
    processed_count = 0
    lines = content.split('\n')

    for table in reversed(tables):  # Process from end to beginning
        try:
            print(f"🤖 Processing markdown table {len(tables) - processed_count}/{len(tables)}...")

            # Parse table structure
            structure = parse_markdown_table_structure(table['table_lines'])
            if not structure:
                continue

            table_data = {
                'structure': structure,
                'raw_content': table['raw_content']
            }

            # Process with Gemini
            sentences = process_markdown_table_with_gemini(table_data, table['context'], product_name)

            if sentences:
                # Replace the table with sentences
                replacement_text = '\n'.join(sentences)

                # Replace the table lines with the processed sentences
                lines[table['start_line']:table['end_line'] + 1] = [replacement_text]

                print(f"   ✅ Converted table to {len(sentences)} standalone sentences")
                processed_count += 1
            else:
                print(f"   ⚠️ No sentences generated for table")

            # Add delay to respect API limits
            logger.debug(f"Applying API delay: {config.api_delay_seconds}s")
            time.sleep(config.api_delay_seconds)

        except Exception as e:
            print(f"   ❌ Error processing table: {str(e)}")
            continue

    # Save the updated content
    if processed_count > 0:
        try:
            updated_content = '\n'.join(lines)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)

            print(f"✅ Updated {file_path} with {processed_count} processed tables")
        except Exception as e:
            print(f"❌ Error saving updated file: {str(e)}")

    return processed_count

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to extract all content from a webpage with enhanced FAQ deduplication"""
    parser = argparse.ArgumentParser(description='Crawling Agent - Enhanced Content Extractor with FAQ Deduplication')
    parser.add_argument('url', help='URL to extract content from')
    args = parser.parse_args()

    url = args.url
    logger.info(f"Starting crawling agent for URL: {url}")

    try:
        print("🤖 Crawling Agent - Enhanced Content Extractor")
    except Exception:
        # Fallback without emoji for Windows cp1252 terminals
        print("Crawling Agent - Enhanced Content Extractor")
    print("=" * 80)
    print(f"🎯 Target URL: {url}")

    # Get product name from URL
    product_name = get_product_name_from_url(url)
    print(f"📦 Product: {product_name}")

    # Create folder structure
    create_folder_structure(product_name)
    print("=" * 80)

    # Get webpage content
    print("🌐 Fetching webpage content...")
    soup = get_webpage_content(url)
    if not soup:
        logger.error("Failed to fetch webpage content")
        print("❌ Failed to fetch webpage content")
        sys.exit(1)

    logger.info("Webpage content loaded successfully")
    print("✅ Webpage content loaded successfully")
    print("=" * 80)

    # Initialize counters
    total_faqs = 0
    total_tables = 0
    total_pdfs = 0
    total_parsed = 0
    total_md_tables = 0

    # Extract FAQs with Enhanced Deduplication
    try:
        logger.info("Starting Phase 1: FAQ Extraction & Deduplication")
        print("📋 PHASE 1: ENHANCED FAQ EXTRACTION & DEDUPLICATION")
        print("-" * 40)
        total_faqs = extract_faqs_enhanced(soup, product_name)
        logger.info(f"Phase 1 completed: {total_faqs} deduplicated FAQs")
        try:
            print(f"✅ Deduplicated FAQs: {total_faqs}")
        except Exception:
            print(f"Deduplicated FAQs: {total_faqs}")
    except Exception as e:
        error_msg = f"Enhanced FAQ extraction failed: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")

    print("=" * 80)

    # Extract Tables
    try:
        logger.info("Starting Phase 2: Table Extraction")
        try:
            print("📊 PHASE 2: TABLE EXTRACTION")
        except Exception:
            print("PHASE 2: TABLE EXTRACTION")
        print("-" * 40)
        total_tables = extract_tables(soup, product_name)
        logger.info(f"Phase 2 completed: {total_tables} table sentences generated")
        try:
            print(f"✅ Table sentences generated: {total_tables}")
        except Exception:
            print(f"Table sentences generated: {total_tables}")
    except Exception as e:
        error_msg = f"Table extraction failed: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")

    print("=" * 80)

    # Extract PDFs
    try:
        logger.info("Starting Phase 3: PDF Extraction")
        try:
            print("📄 PHASE 3: PDF EXTRACTION")
        except Exception:
            print("PHASE 3: PDF EXTRACTION")
        print("-" * 40)
        downloaded_files = extract_pdfs(soup, url, product_name)
        total_pdfs = len(downloaded_files)
        logger.info(f"Phase 3 completed: {total_pdfs} PDFs downloaded")
        try:
            print(f"✅ PDFs downloaded: {total_pdfs}")
        except Exception:
            print(f"PDFs downloaded: {total_pdfs}")
    except Exception as e:
        error_msg = f"PDF extraction failed: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        downloaded_files = []

    print("=" * 80)

    # Parse PDFs with LlamaParse
    try:
        logger.info("Starting Phase 4: PDF Parsing with LlamaParse")
        try:
            print("🤖 PHASE 4: PDF PARSING WITH LLAMAPARSE")
        except Exception:
            print("PHASE 4: PDF PARSING WITH LLAMAPARSE")
        print("-" * 40)
        total_parsed = parse_pdfs_with_llamaparse(product_name, downloaded_files)
        logger.info(f"Phase 4 completed: {total_parsed} PDFs parsed")
        try:
            print(f"✅ PDFs parsed: {total_parsed}")
        except Exception:
            print(f"PDFs parsed: {total_parsed}")
    except Exception as e:
        error_msg = f"PDF parsing failed: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")

    print("=" * 80)

    # Process markdown tables in parsed content
    try:
        logger.info("Starting Phase 5: Markdown Table Processing")
        try:
            print("📊 PHASE 5: MARKDOWN TABLE PROCESSING")
        except Exception:
            print("PHASE 5: MARKDOWN TABLE PROCESSING")
        print("-" * 40)
        parsed_file = f"source_db/policy/{product_name}_policy.md"
        total_md_tables = process_markdown_tables_in_file(parsed_file, product_name)
        logger.info(f"Phase 5 completed: {total_md_tables} markdown tables processed")
        try:
            print(f"✅ Markdown tables processed: {total_md_tables}")
        except Exception:
            print(f"Markdown tables processed: {total_md_tables}")
    except Exception as e:
        error_msg = f"Markdown table processing failed: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")

    

    # Final Summary
    logger.info("Generating final summary")
    print("=" * 80)
    try:
        print("📈 CRAWLING AGENT SUMMARY")
    except Exception:
        print("CRAWLING AGENT SUMMARY")
    print("=" * 80)
    print(f"Product: {product_name}")
    print(f"URL: {url}")
    print(f"Deduplicated FAQs: {total_faqs}")
    print(f"Table sentences: {total_tables}")
    print(f"PDFs downloaded: {total_pdfs}")
    print(f"PDFs parsed: {total_parsed}")
    print(f"Markdown tables processed: {total_md_tables}")
    print("=" * 80)

    # Log summary statistics
    logger.info(f"Crawling completed for {product_name}: FAQs={total_faqs}, Tables={total_tables}, PDFs={total_pdfs}, Parsed={total_parsed}, MD_Tables={total_md_tables}")

    # List created files
    print("📁 FILES CREATED:")
    print("-" * 40)

    # FAQs
    faq_file = f"source_db/FAQ/{product_name}_FAQs.txt"
    if os.path.exists(faq_file):
        size = os.path.getsize(faq_file)
        print(f"📋 {faq_file} ({size:,} bytes) - DEDUPLICATED")

    # Benefits (Tables)
    benefits_file = f"source_db/benefits/{product_name}_benefits.txt"
    if os.path.exists(benefits_file):
        size = os.path.getsize(benefits_file)
        print(f"📊 {benefits_file} ({size:,} bytes)")

    # PDFs
    pdf_folder = f"source_db/pdfs/{product_name}"
    if os.path.exists(pdf_folder):
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        print(f"📄 PDF folder: {len(pdf_files)} files in {pdf_folder}/")

    # Policy (Parsed content)
    policy_file = f"source_db/policy/{product_name}_policy.md"
    if os.path.exists(policy_file):
        size = os.path.getsize(policy_file)
        print(f"🤖 {policy_file} ({size:,} bytes) - TABLES PROCESSED")

    print("=" * 80)

    if total_faqs + total_tables + total_pdfs + total_parsed + total_md_tables > 0:
        logger.info("Crawling Agent completed successfully")
        try:
            print("✅ Crawling Agent completed successfully!")
        except Exception:
            print("Crawling Agent completed successfully!")
        print("💾 All content saved with enhanced deduplication")
        print("🔍 FAQs have been deduplicated across all sources")
        if total_parsed > 0:
            try:
                print("🤖 PDF content parsed and ready for RAG system")
            except Exception:
                print("PDF content parsed and ready for RAG system")
        if total_md_tables > 0:
            try:
                print("📊 Markdown tables converted to standalone sentences")
            except Exception:
                print("Markdown tables converted to standalone sentences")
    else:
        logger.warning("No content was extracted from the webpage")
        print("⚠️ No content was extracted from the webpage")
        print("💡 The webpage might not contain the expected content types")


if __name__ == "__main__":
    main()
