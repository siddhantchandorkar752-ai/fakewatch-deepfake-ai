from setuptools import setup, find_packages

setup(
    name="fakewatch",
    version="2.0.0",
    description="FAKEWATCH — Deepfake Forensic Intelligence System",
    author="Siddhant Chandorkar",
    author_email="siddhantchandorkar752@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "torchvision",
        "opencv-python",
        "ultralytics",
        "timm",
        "scikit-learn",
        "pyyaml",
        "fastapi",
        "uvicorn",
        "gradio",
        "numpy",
        "pandas",
        "python-multipart",
        "onnx",
        "onnxruntime",
        "python-dotenv",
    ],
)
