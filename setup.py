from setuptools import setup


PACKAGES = [
    "agentic_mm_rag",
    "agentic_mm_rag.agent",
    "agentic_mm_rag.data",
    "agentic_mm_rag.orchestrator",
    "agentic_mm_rag.orchestrator.evidence",
    "agentic_mm_rag.providers",
    "agentic_mm_rag.tools",
    "agentic_mm_rag.tools.runtime",
    "agentic_mm_rag.tools.runtime.stores",
]


setup(
    name="agentic-mm-rag",
    version="0.1.0",
    description="Storage-native agentic multimodal RAG for documents, videos, and fused evidence.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    license="MIT",
    author="Agentic MM-RAG contributors",
    package_dir={"agentic_mm_rag": "."},
    packages=PACKAGES,
    package_data={
        "agentic_mm_rag": [
            "data/README.md",
            "data/ingestion.env.example",
        ]
    },
    include_package_data=True,
    install_requires=[],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "dev": ["pytest>=8.0"],
        "doc-ingest": ["lightrag-hku"],
        "video-ingest": ["moviepy", "pillow", "numpy"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
