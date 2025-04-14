from setuptools import setup, find_packages

setup(
    name="gOCR",
    version="0.0.2",
    packages=find_packages(),
    install_requires=[
        'requests>=2.25.1',
        'pdfplumber>=0.7.0',
        'pdf2image>=1.16.3',
        'Pillow>=10.0.0',
    ],
    author="SamIppp",
    author_email="i60996395@gmail.com",
    description="OCR and PDF processing utilities",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SamIp-ac/gOCR",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)