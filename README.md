# gOCR

OCR and PDF processing utilities

## Installation

Remarks: default deploy on local lm studio / cli

```bash
pip install git+https://github.com/SamIp-ac/gOCR.git
```

 ## Usage Example:
 ```python
 from gOCR import gOCR

# Initialize with custom settings
processor = gOCR(
    llm_host="http://your-llm-server:1234",
    api_key="your-api-key-if-needed"
)

# Process PDF
pdf_text = processor.load_pdf("document.pdf")

# Get LLM completion
response = processor.call_chat_completion(
    ocr_prompt=pdf_text,
    system_prompt="You are a helpful assistant.",
    user_prompt="Summarize this document",
    model="gpt-4"  # Override default model
)

print(response)
```

## For testing
```shell
conda create -n gOCR_py312 python=3.12
conda activate gOCR_py312
```

conda install -c conda-forge poppler
pip install -e .

Make sure you have poppler installed on your system (required by pdf2image):
On macOS: brew install poppler
On Ubuntu: apt-get install poppler-utils
On Windows: Download and install poppler binaries

ocr = gOCR(llm_host="your_llm_host")
result = ocr.process_pdf_with_images(
    pdf_path="your.pdf",
    system_prompt="Analyze this document",
    user_prompt="Extract all the information from these images"
)