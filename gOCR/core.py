import requests
import pdfplumber
from typing import Optional, List, Dict, Union
import pdf2image
import base64
from io import BytesIO
from PIL import Image

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

    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[Image.Image]:
        """
        Convert PDF pages to PIL Image objects.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: DPI for the output images (default: 200)
            
        Returns:
            List of PIL Image objects, one for each page
        """
        try:
            return pdf2image.convert_from_path(pdf_path, dpi=dpi)
        except Exception as e:
            print(f"Failed to convert PDF to images: {e}")
            return []

    def image_to_base64(self, image: Image.Image, format: str = 'JPEG') -> str:
        """
        Convert a PIL Image to base64 string.
        
        Args:
            image: PIL Image object
            format: Output format (default: 'JPEG')
            
        Returns:
            Base64 encoded string of the image
        """
        buffered = BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode()

    def call_chat_completion_with_images(
        self,
        system_prompt: str,
        user_prompt: str,
        images: List[Image.Image],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Optional[str]:
        """
        Calls the chat completion API with images.
        
        Args:
            system_prompt: The system instruction/context
            user_prompt: The user's question/request
            images: List of PIL Image objects to process
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

        # Convert images to base64 and create content list
        content = []
        for img in images:
            encoded_image = self.image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            })
        
        # Add the text prompt at the end
        content.append({"type": "text", "text": user_prompt})

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": content
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

    def process_pdf_with_images(
        self,
        pdf_path: str,
        system_prompt: str,
        user_prompt: str,
        dpi: int = 200,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Optional[str]:
        """
        Process a PDF file by converting it to images and sending to LLM.
        
        Args:
            pdf_path: Path to the PDF file
            system_prompt: The system instruction/context
            user_prompt: The user's question/request
            dpi: DPI for the output images (default: 200)
            model: The model to use (default: from initialization)
            max_tokens: Maximum tokens to generate (default: from initialization)
            temperature: Sampling temperature (default: from initialization)
            
        Returns:
            The generated response or None if failed
        """
        images = self.pdf_to_images(pdf_path, dpi)
        if not images:
            return None
            
        return self.call_chat_completion_with_images(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def call_chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        ocr_prompt: Optional[str] = "",
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
            text_content = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text_content.append(page.extract_text())
            return "\n".join(text_content)
        except Exception as e:
            print(f"Failed to load PDF: {e}")
            return ""