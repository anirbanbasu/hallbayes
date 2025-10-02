from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hallbayes",
    version="0.1.0",
    author="Leo",
    description="EDFL/B2T/ISR hallucination risk calculator for LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leochlon/hallbayes",
    packages=find_packages(include=["hallbayes*", "app*"]),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "numpy",
        "requests",
    ],
    extras_require={
        "anthropic": ["anthropic>=0.28.0"],
        "huggingface": ["transformers", "torch"],
        "ollama": ["ollama"],
        "openrouter": ["requests"],  # OpenRouter just needs requests
        "all": ["anthropic>=0.28.0", "transformers", "torch", "ollama", "requests"],
    },
)
