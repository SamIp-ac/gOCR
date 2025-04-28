# gOCR

OCR and PDF processing utilities

## Installation

Remarks: default deploy on local lm studio / cli, "Try mmap()" -> false

```bash
pip install git+https://github.com/SamIp-ac/gOCR.git
```
 ## Usage Example:
 ```python
 from gOCR import gOCR
# Example Usage:
LLM_HOST = "http://localhost:1234"  # Update this with your LLM host
API_KEY = None  # Add your API key if needed
ocr_processor = gOCR(llm_host=LLM_HOST, api_key=API_KEY)
#
# Example 1: Process a PDF using combined text + image approach
result_combined = ocr_processor.process_pdf_combined(
    model="gemma-3-12b-it",
    max_tokens=2048,
    temperature=.1,
    pdf_path="gOCR/assets/template.pdf",
    system_prompt="You are an expert data extractor.",
    user_prompt="Extract the number, total amount, and date.",
)
print("Combined Processing Result:")
print(result_combined)

# Example 2: Just load all text using LangChain loader (static method)
all_text = gOCR.load_pdf("/path/to/your/document.pdf")
print("\nFull Text (via load_pdf):")
print(all_text[:500] + "...") # Print first 500 chars

# Example 3: Just load first page text using LangChain loader (static method)
first_page_text = gOCR.load_pdf_first_page("/path/to/your/document.pdf")
print("\nFirst Page Text (via load_pdf_first_page):")
print(first_page_text)
```

## For testing
```shell
conda create -n gOCR_py312 python=3.12
conda activate gOCR_py312

pip3 install -e .
```

## Run api
```shell
pip3 install fastapi uvicorn python-multipart
python main.py
```
## streamlit run app.py
```shell
pip install streamlit requests
streamlit run app.py
```

## Debug for zbar not found even already installed
```shell
mkdir ~/lib
ln -s $(brew --prefix zbar)/lib/libzbar.dylib ~/lib/libzbar.dylib
```


Make sure you have poppler installed on your system (required by pdf2image):
On macOS: brew install poppler
On Ubuntu: apt-get install poppler-utils
On Windows: Download and install poppler binaries
