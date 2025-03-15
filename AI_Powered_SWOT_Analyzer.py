#!/usr/bin/env python
# coding: utf-8

# ### Objectives:
# 
# 1. Develop an AI-driven SWOT Analysis Agent using Streamlit and LangChain to analyze business information and generate insights.
# 2. Provide a detailed SWOT breakdown that includes insights.
# 3. Offer business recommendations based on the SWOT findings to support strategic decision-making.

# ### Importing Libraries

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import os # For interacting with the operating system (e.g., accessing environment variables).
import streamlit as st # For building the web application interface.
import langchain # A framework for building applications using LLMs.
import transformers # For working with transformer models and tokenizers.
from langchain_google_genai import GoogleGenerativeAI # For interfacing with Google's Gemini models.
from langchain.prompts import PromptTemplate # For creating prompt templates for the LLM.
from langchain.chains import LLMChain # For creating chains that connect prompts and LLMs.
from langchain.chains import SimpleSequentialChain # For creating simple sequential chains of LLM calls.
import re # For using regular expressions to parse LLM output.
from transformers import AutoTokenizer # For tokenizing text using pre-trained models.
from langchain.text_splitter import CharacterTextSplitter # For splitting long texts into smaller chunks.
import PyPDF2 # For extracting text from PDF files.
from io import BytesIO  # Import BytesIO - For handling PDF files in memory.


# ### Printing the Versions of Libraries Used

# In[3]:


# Print library versions
print("Libraries used:")
print(f"- streamlit: {st.__version__}")
print(f"- langchain: {langchain.__version__}")
print(f"- transformers: {transformers.__version__}")
print(f"- PyPDF2: {PyPDF2.__version__}")
print("- re") # Built-in module, no version
print("- os") # Built-in module, no version
print("- io") #Built-in module, no version


# In[ ]:


# Google Generative AI (langchain_google_genai) version: Version information not available


# ### Initial Steps

# In[ ]:


# --- Streamlit App Configuration ---
st.set_page_config(page_title="AI-Powered SWOT Analyzer") # Sets the page title in the browser tab.

# Load API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Retrieves the Google API key from the environment variables.
if not GOOGLE_API_KEY:
    st.error("Google API Key is missing! Set the GEMINI_API_KEY environment variable.")
    st.stop()

# --- Initialize Gemini Model ---
gemini_llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=GOOGLE_API_KEY) # Initializes the Gemini LLM with specified parameters.

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file): # Defines a function to extract text from PDF files.
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Process long text by splitting and summarizing
def process_long_text(text): # Defines a function to process long texts by splitting and summarizing.
    text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # For multiple chunks, summarize first
    if len(chunks) > 1:
        model = get_model()
        summaries = []
        for chunk in chunks:
            response = model.generate_content(f"Summarize the following text concisely:\n\n{chunk}")
            summaries.append(response.text)
        
        return " ".join(summaries)
    else:
        return text


# ### Creating the Prompt Templates

# In[ ]:


# --- Prompt Templates ---
swot_prompt_template = """
You are an expert strategic analyst specializing in SWOT analysis. Analyze the following text and identify the Strengths, Weaknesses, Opportunities, and Threats. Format each item as a numbered list (1, 2, 3, ...).

Text:
{text}

SWOT Analysis:
""" # Defines a prompt template for SWOT analysis.
interpretation_prompt_template = """
You are an expert business strategist. Given the following SWOT analysis, provide an interpretation of the key strategic implications. Focus on how the strengths can be used to capitalize on opportunities, and how weaknesses can be mitigated to avoid threats. Do not refer to the SWOT items using numbers. Just discuss them.

SWOT Analysis:
{swot_analysis}

Interpretation:
""" # Defines a prompt template for interpreting the SWOT analysis.
recommendation_prompt_template = """
You are a seasoned business consultant. Based on the following SWOT analysis and its interpretation, provide three high-level strategic recommendations. These recommendations should be actionable and aimed at improving the overall business performance.  Do not refer to the SWOT items using numbers. Just discuss them.

Combined Analysis:
{combined_analysis}

Recommendations:
""" # Defines a prompt template for generating strategic recommendations.

swot_prompt = PromptTemplate(input_variables=["text"], template=swot_prompt_template) # Creates a PromptTemplate instance for SWOT analysis.
interpretation_prompt = PromptTemplate(input_variables=["swot_analysis"], template=interpretation_prompt_template) # Creates a PromptTemplate instance for interpretation.
recommendation_prompt = PromptTemplate(input_variables=["combined_analysis"], template=recommendation_prompt_template) # Creates a PromptTemplate instance for recommendations.

swot_chain = LLMChain(prompt=swot_prompt, llm=gemini_llm, output_key="swot_analysis") # Creates an LLMChain for SWOT analysis.
interpretation_chain = LLMChain(prompt=interpretation_prompt, llm=gemini_llm, output_key="interpretation") # Creates an LLMChain for interpretation.
recommendation_chain = LLMChain(prompt=recommendation_prompt, llm=gemini_llm, output_key="recommendations") # Creates an LLMChain for recommendations.

overall_chain = SimpleSequentialChain(chains=[swot_chain, interpretation_chain], verbose=False) # Creates a sequential chain for SWOT analysis and interpretation.


# ### Creating Token Tracker and Setting Colour Palette for UI

# In[ ]:


# --- Token Tracking Variables ---
input_tokens_used = 0 # Initializes a variable to track input tokens used.
output_tokens_used = 0 # Initializes a variable to track output tokens used.

# --- Define Color Palette ---
STRENGTH_COLOR = "#77DD77"  # Light Green
WEAKNESS_COLOR = "#FF6961"  # Light Red
OPPORTUNITY_COLOR = "#FFB347" # Light Orange
THREAT_COLOR = "#AEC6CF"    # Light Blue

# Initialize the tokenizer (This is important!)
try:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")  # Replace with a more suitable Gemini tokenizer if one exists - Initializes the tokenizer for counting tokens.
except Exception as e:
    st.warning(f"Failed to load tokenizer: {e}. Token counts will be a rough estimate.")
    tokenizer = None

def count_tokens(text: str, tokenizer) -> int: # Defines a function to count tokens in a text string.
    """Counts the number of tokens in a text string using a Hugging Face tokenizer."""
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            st.warning(f"Failed to count tokens using tokenizer: {e}. Using a rough estimate (word count).")
            return len(text.split())  # Fallback to a rough estimate if tokenization fails
    else:
        return len(text.split())  # Fallback to a rough estimate if no tokenizer is available

def update_token_display(input_used=None, output_used=None): # Defines a function to update the token usage display in the sidebar.
    """Updates the token usage display in the sidebar.  Handles cases where one or both values are not available yet."""
    markdown_text = ""
    if input_used is not None:
        markdown_text += f"""**Input Tokens Used:** {input_used:,}  \n"""
    if output_used is not None:
        markdown_text += f"""**Output Tokens Used:** {output_used:,}"""

    st.sidebar.markdown(markdown_text, unsafe_allow_html=True)


# Initial Token Display (only show input initially, if available)
update_token_display() # Calls the update_token_display function to initially show the token usage.


# ### Setting Up the UI

# In[ ]:


# --- Streamlit App Layout ---
st.title("AI-Powered SWOT Analyzer") # Sets the title of the Streamlit application.
st.markdown("Objective: An AI-powered SWOT Analysis Agent using Streamlit and LangChain to analyze business information, generates actionable insights, provides detailed text and visual SWOT breakdowns, and offers strategic recommendations for decision-making.")

st.markdown("Powered by Gemini-2.0-Flash, LangChain version: 0.3.20, & Streamlit version: 1.37.1")
st.markdown("Transformers version: 4.49.0, Google Generative AI version: Version information not available")
st.markdown("Upload a text file or enter text to generate a SWOT analysis, interpretation, and recommendations.")

uploaded_file = st.file_uploader("Choose a text file", type=["txt", "pdf"]) # Creates a file uploader for text or PDF files.
text_input = st.text_area("Or enter text directly:", height=200) # Creates a text area for direct text input.


# ### Creation of the SWOT Function

# In[ ]:


# Modified file handling to handle PDF uploads correctly
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        # Use BytesIO to handle the PDF file
        text = extract_text_from_pdf(BytesIO(uploaded_file.read()))
    else:
        # For text files, decode as UTF-8
        try:
            text = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            st.error("Error: The uploaded text file is not UTF-8 encoded. Please ensure the file is properly encoded.")
            text = None  # Set text to None to prevent further processing
else:
    text = text_input if text_input else None

if text:
    st.subheader("Original Text:") # Displays the original text input.
    st.markdown(f'<div style="height: 200px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;">{text}</div>', unsafe_allow_html=True)

    if st.button("Analyze"): # Creates an "Analyze" button.
        with st.spinner("Analyzing..."):
            try:
                # Track Input Tokens
                input_tokens_used = count_tokens(text, tokenizer)
                update_token_display(input_used=input_tokens_used) # Only show input tokens

                swot_analysis = swot_chain.run(text)
                interpretation = interpretation_chain.run(swot_analysis=swot_analysis)
                combined_analysis = f"SWOT Analysis:\n{swot_analysis}\n\nInterpretation:\n{interpretation}"
                recommendations = recommendation_chain.run(combined_analysis=combined_analysis)

                # Track Output Tokens
                output_tokens_used = count_tokens(swot_analysis + interpretation + recommendations, tokenizer)
                update_token_display(input_used=input_tokens_used, output_used=output_tokens_used) #Show both

                def extract_swot(text):
                    strengths = re.search(r"strengths:\s*\n?([\s\S]*?)(?=weaknesses:|$)", text, re.IGNORECASE)
                    weaknesses = re.search(r"weaknesses:\s*\n?([\s\S]*?)(?=opportunities:|$)", text, re.IGNORECASE)
                    opportunities = re.search(r"opportunities:\s*\n?([\s\S]*?)(?=threats:|$)", text, re.IGNORECASE)
                    threats = re.search(r"threats:\s*\n?([\s\S]*?)(?=(This SWOT analysis)|$)", text, re.IGNORECASE)
                    summary_match = re.search(r"(This SWOT analysis[\s\S]*)", text, re.IGNORECASE)

                    return {
                        "strengths": strengths.group(1).strip().replace("*", "") if strengths else "",
                        "weaknesses": weaknesses.group(1).strip().replace("*", "") if weaknesses else "",
                        "opportunities": opportunities.group(1).strip().replace("*", "") if opportunities else "",
                        "threats": threats.group(1).strip().replace("*", "") if threats else "",
                        "summary": summary_match.group(1).strip() if summary_match else ""
                    }

                swot_data = extract_swot(swot_analysis)

                # Collapsible Sections
                with st.expander("SWOT Analysis"): # Creates a collapsible section for the SWOT analysis results.
                    col1, col2 = st.columns(2)
                    col3, col4 = st.columns(2)

                    with col1:
                        st.markdown(f"<h3 style='color: white; background-color: {STRENGTH_COLOR}; padding: 10px; border-radius: 5px; text-align: center;'>Strengths</h3>", unsafe_allow_html=True)
                        st.write(swot_data["strengths"])
                    with col2:
                        st.markdown(f"<h3 style='color: white; background-color: {WEAKNESS_COLOR}; padding: 10px; border-radius: 5px; text-align: center;'>Weaknesses</h3>", unsafe_allow_html=True)
                        st.write(swot_data["weaknesses"])
                    with col3:
                        st.markdown(f"<h3 style='color: white; background-color: {OPPORTUNITY_COLOR}; padding: 10px; border-radius: 5px; text-align: center;'>Opportunities</h3>", unsafe_allow_html=True)
                        st.write(swot_data["opportunities"])
                    with col4:
                        st.markdown(f"<h3 style='color: white; background-color: {THREAT_COLOR}; padding: 10px; border-radius: 5px; text-align: center;'>Threats</h3>", unsafe_allow_html=True)
                        st.write(swot_data["threats"])

                with st.expander("Interpretation"): # Creates a collapsible section for the interpretation.
                    st.write(interpretation)

                with st.expander("Recommendations"): # Creates a collapsible section for the recommendations.
                    st.write(recommendations)

            except Exception as e:
                st.error(f"Error: {e}")

