# gOCR

OCR and PDF processing utilities

## Installation

```bash
pip install git+https://github.com/yourusername/gOCR.git
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