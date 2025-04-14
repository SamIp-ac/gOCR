import requests
from pdfplumber import PDFPlumberLoader
from typing import Optional

class gOCR:
    def __init__(self, llm_host: str = "http://localhost:1234", api_key: Optional[str] = None):
        """
        Initialize the gOCR processor with LLM connection settings.
        
        Args:
            llm_host: Base URL for the LLM API (default: "http://localhost:1234")
            api_key: Optional API key for authentication
        """
        self.llm_host = llm_host.rstrip('/')  # Remove trailing slash if present
        self.api_key = api_key
        self.default_model = "gemma-3-4b-it"
        self.default_max_tokens = 4000
        self.default_temperature = 0.1

    def call_chat_completion(
        self,
        ocr_prompt: str,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Optional[str]:
        """
        Calls the chat completion API with OCR, system, and user prompts.
        
        Args:
            ocr_prompt: The text extracted from OCR
            system_prompt: The system instruction/context
            user_prompt: The user's question/request
            model: The model to use (default: from initialization)
            max_tokens: Maximum tokens to generate (default: from initialization)
            temperature: Sampling temperature (default: from initialization)
            
        Returns:
            The generated response or None if failed
        """
        url = f"{self.llm_host}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ocr_prompt + "\n" + user_prompt},
                ]
            }
        ]

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature or self.default_temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except (KeyError, ValueError) as e:
            print(f"Failed to parse response: {e}")
            return None

    @staticmethod
    def load_pdf(file_path: str) -> str:
        """
        Static method to load and extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Combined text from all pages of the PDF
        """
        try:
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Failed to load PDF: {e}")
            return ""