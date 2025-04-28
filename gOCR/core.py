import requests
# import pdfplumber # No longer directly used in static methods, but pdf2image might need it installed
from typing import Optional, List, Dict, Union
import pdf2image
import base64
from io import BytesIO
from PIL import Image
import re
import os
# import tempfile # Not strictly needed if we use the path directly

# --- Import the required LangChain Loader ---
from langchain_community.document_loaders import PDFPlumberLoader
# ---

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
        self.default_model = "gemma-3-12b-it"
        self.default_max_tokens = 7500
        self.default_temperature = 0.1

    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[Image.Image]:
        """
        Convert PDF pages to PIL Image objects. (Requires pdf2image and poppler)

        Args:
            pdf_path: Path to the PDF file
            dpi: DPI for the output images (default: 200)

        Returns:
            List of PIL Image objects, one for each page
        """
        try:
            return pdf2image.convert_from_path(pdf_path, dpi=dpi)
        except pdf2image.exceptions.PDFInfoNotInstalledError:
             print("ERROR: pdf_to_images failed: poppler (pdfinfo) not found or not in PATH.")
             print("Please install poppler-utils (Linux/macOS) or Poppler for Windows.")
             return []
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
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                 print(f"Response status code: {e.response.status_code}")
                 print(f"Response text: {e.response.text}")
            return None
        except (KeyError, IndexError, ValueError) as e:
            print(f"Failed to parse response or access data: {e}")
            # Optionally log detailed error: print(response_data)
            return None

    def process_pdf_with_images(
        self,
        pdf_paths: Union[str, List[str]],
        system_prompt: str,
        user_prompt: str,
        dpi: int = 200,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Optional[str]:
        """
        Process one or multiple PDF files by converting them to images and sending to LLM.

        Args:
            pdf_paths: Path to a single PDF file or list of PDF file paths
            system_prompt: The system instruction/context
            user_prompt: The user's question/request
            dpi: DPI for the output images (default: 200)
            model: The model to use (default: from initialization)
            max_tokens: Maximum tokens to generate (default: from initialization)
            temperature: Sampling temperature (default: from initialization)

        Returns:
            The generated response or None if failed
        """
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]

        all_images = []
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"Warning: PDF file not found at {pdf_path}, skipping.")
                continue
            images = self.pdf_to_images(pdf_path, dpi) # Calls method that handles poppler error
            if images:
                all_images.extend(images)
            elif not images: # Explicitly check if list is empty after call
                 print(f"Warning: No images generated from {pdf_path}. Check poppler installation and PDF validity.")
                 # Decide if you want to continue without images for this PDF or fail
                 # continue # Skip this PDF if images are crucial

        if not all_images:
            print("Error: No images could be extracted from any of the provided PDF paths.")
            return None # Fail if no images were generated at all

        return self.call_chat_completion_with_images(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=all_images,
            model=model or self.default_model,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            temperature=temperature if temperature is not None else self.default_temperature
        )

    # --- Method using combined text (1st page) + images ---
    def process_pdf_combined(
        self,
        pdf_path: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None
    ) -> Dict[str, Union[str, Dict]]:
        """
        Processes a single PDF file using text extraction from the first page
        (via LangChain PDFPlumberLoader) and image-based processing for the whole
        document, combining the info for the LLM call.

        Args:
            pdf_path: Path to the PDF file to process.
            system_prompt: The system instruction/context for the LLM.
            user_prompt: The user's specific question/request for the LLM.
                         The extracted text from the first page will be prepended.
            max_tokens: Maximum tokens for LLM response (uses class default if None).
            temperature: Temperature for LLM response (uses class default if None).
            model: Model name for the LLM call (uses class default if None).

        Returns:
            A dictionary containing either {"result": <llm_response>} on success
            or {"error": <error_message>} on failure.
        """
        temp_path_used = None # Variable to track temporary file if needed
        try:
            # --- File Handling ---
            # If pdf_path is an UploadFile object (from FastAPI context), save it temporarily.
            # Otherwise, assume it's already a valid path string.
            # This check makes the method more versatile.
            if hasattr(pdf_path, 'file') and hasattr(pdf_path, 'filename'): # Check if it looks like FastAPI UploadFile
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    content = pdf_path.file.read() # Read from UploadFile's file object
                    temp_file.write(content)
                    current_pdf_path = temp_file.name
                    temp_path_used = current_pdf_path # Remember path for cleanup
                    pdf_path.file.seek(0) # Reset file pointer in case it's used again
                    print(f"Processing temporary file: {current_pdf_path}")
            elif isinstance(pdf_path, str) and os.path.exists(pdf_path):
                 current_pdf_path = pdf_path
                 print(f"Processing provided path: {current_pdf_path}")
            else:
                 return {"error": f"Input PDF path is invalid or file not found: {pdf_path}"}


            # --- Extract text from the first page using the static method ---
            plumber_ocr = self.load_pdf_first_page(current_pdf_path) # Use potentially temporary path
            if not plumber_ocr:
                 print(f"Warning: Could not extract text from the first page of {current_pdf_path} using LangChain loader.")
                 plumber_ocr = "" # Ensure it's an empty string if extraction failed

            # --- Define final parameters using method args or class defaults ---
            final_model = model if model is not None else self.default_model
            final_max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
            final_temperature = temperature if temperature is not None else self.default_temperature

            # --- Construct the user prompt including the first-page text ---
            combined_user_prompt = (
                f"Here is the text extracted from the first page for context:\n--- START FIRST PAGE TEXT ---\n"
                f"{plumber_ocr}\n"
                f"--- END FIRST PAGE TEXT ---\n\n"
                f"Now, considering the entire document (including images), please address the following:\n"
                f"{user_prompt}"
            )

            # --- Process the entire PDF using the image-based method ---
            response = self.process_pdf_with_images(
                pdf_paths=[current_pdf_path], # Use potentially temporary path
                system_prompt=system_prompt,
                user_prompt=combined_user_prompt,
                model=final_model,
                max_tokens=final_max_tokens,
                temperature=final_temperature
            )

            # --- Check the response from the LLM call ---
            if response is None:
                # Error likely already printed. Add context.
                return {"error": "Failed to process PDF with image processing (LLM call failed or returned no content, or image generation failed)"}

            # Successfully processed
            return {"result": response}

        except Exception as e:
            import traceback
            print(f"An unexpected error occurred in process_pdf_combined: {traceback.format_exc()}")
            return {"error": f"An unexpected error occurred: {str(e)}"}
        finally:
             # --- Clean up temporary file if one was created ---
             if temp_path_used and os.path.exists(temp_path_used):
                 try:
                     os.unlink(temp_path_used)
                     print(f"Cleaned up temporary file: {temp_path_used}")
                 except OSError as e:
                     print(f"Error cleaning up temporary file {temp_path_used}: {e}")


    # --- Existing text-only call_chat_completion (kept for potential other uses) ---
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
        Calls the chat completion API with OCR, system, and user prompts. TEXT ONLY.

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
                    {"type": "text", "text": (ocr_prompt + "\n" + user_prompt).strip()}, # Combine prompts
                ]
            }
        ]

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                 print(f"Response status code: {e.response.status_code}")
                 print(f"Response text: {e.response.text}")
            return None
        except (KeyError, IndexError, ValueError) as e:
            print(f"Failed to parse response or access data: {e}")
            # Optionally log detailed error: print(response_data)
            return None

    # --- Static methods for PDF text extraction using LangChain ---
    @staticmethod
    def _clean_text(raw_text: str) -> str:
        """Internal helper to clean extracted text."""
        if not raw_text:
            return ""
        # Clean up CID patterns
        cleaned_text = re.sub(r'\(cid:\d+\)', '', raw_text)
        # Optional: Replace common ligatures if needed
        # cleaned_text = cleaned_text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        # Normalize whitespace (replace multiple spaces/newlines with single space, then strip)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    @staticmethod
    def load_pdf(file_path: str) -> str:
        """
        Static method to load and extract text from all pages of a PDF file
        using LangChain's PDFPlumberLoader. Cleans up extracted text.

        Args:
            file_path: Path to the PDF file

        Returns:
            Combined and cleaned text from all pages, or "" on error.
        """
        if not os.path.exists(file_path):
            print(f"Error in load_pdf: File not found at {file_path}")
            return ""
        try:
            # Use PDFPlumberLoader from LangChain
            loader = PDFPlumberLoader(file_path)
            docs = loader.load() # Returns a list of Document objects

            if not docs:
                 print(f"Warning: LangChain PDFPlumberLoader returned no documents for {file_path}")
                 return ""

            # Combine page content and clean it
            all_text = "\n".join([gOCR._clean_text(doc.page_content) for doc in docs if doc.page_content])
            return all_text

        except Exception as e:
            print(f"Failed to load PDF '{file_path}' using LangChain PDFPlumberLoader: {e}")
            import traceback
            print(traceback.format_exc())
            return ""

    @staticmethod
    def load_pdf_first_page(file_path: str) -> str:
        """
        Static method to load and extract text from the first page of a PDF file
        using LangChain's PDFPlumberLoader. Cleans up extracted text.

        Args:
            file_path: Path to the PDF file

        Returns:
            Cleaned text from the first page, or "" on error/no text.
        """
        if not os.path.exists(file_path):
            print(f"Error in load_pdf_first_page: File not found at {file_path}")
            return ""
        try:
            # Use PDFPlumberLoader from LangChain
            loader = PDFPlumberLoader(file_path)
            docs = loader.load() # Loads all pages

            if not docs:
                print(f"Warning: LangChain PDFPlumberLoader returned no documents for {file_path}")
                return ""

            # Get content from the first document (first page) and clean it
            first_page_raw_text = docs[0].page_content
            return gOCR._clean_text(first_page_raw_text)

        except Exception as e:
            print(f"Failed to load first page of PDF '{file_path}' using LangChain PDFPlumberLoader: {e}")
            import traceback
            print(traceback.format_exc())
            return ""
