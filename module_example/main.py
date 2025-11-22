import argparse
import asyncio
import os
from typing import Any, Dict

import dspy
from dspy import asyncify
from attachments.dspy import Attachments
from modules.DocumentClassifier import DocumentClassifier, DocumentType, SECDocumentType
from modules.VisualWithTools import VisualAnalyzerWithTools
from modules.BoundaryDetector import PDFBoundaryDetector
from modules.DocumentProcessor import Form4Processor
from modules.Summarizer import Summarizer
from pprint import pprint


async def classify_file(file_path: str) -> str:
    """Classify a file into a type."""
    classifier = asyncify(DocumentClassifier(file_path))
    return await classifier()


async def process_single_file(file_path: str) -> Dict[str, Any]:
    """Process a single file: classify and then analyze with appropriate module."""
    doc = Attachments(file_path)
    file_type: DocumentType = await classify_file(file_path)

    print(f"File type: {file_type.document_type}")

    if file_type.document_type == "SEC_FILING":
        # Get the SEC filing type using a smaller model
        with dspy.context(lm=lm_small):
            sec_type = dspy.Predict(SECDocumentType)(
                document_images=dspy.Image(doc.images[0])
            )
            print(f"SEC filing type: {sec_type.document_type}")

        if sec_type.document_type == "FORM 4":
            data = Form4Processor()(doc=doc)
            assert data.filing_date == "2025-09-15"
            assert data.total_shares_sold == 497797

            return data

    elif file_type.document_type == "CONTRACT":
        # Run summarization and boundary detection concurrently
        summarizer = asyncify(Summarizer(file_path))
        detector = PDFBoundaryDetector(file_path)

        summary, boundaries = await asyncio.gather(summarizer(), detector.aforward())

        print(summary)
        print("-" * 100)
        print(boundaries)

        return None

    elif file_type.document_type == "CITY_INFRASTRUCTURE":
        # load file as bytes
        with open(file_path, "rb") as f:
            image_bytes = f.read()

        # OPTION 1 - hard coded
        # visual_analyzer = VisualAnalyzer()
        # res = visual_analyzer(image_bytes=image_bytes)

        # OPTION 2 - use LM with tools
        # Use specific LM here
        lm_gemini = dspy.LM(
            model="openrouter/google/gemini-2.5-flash",
            # model="openrouter/google/gemini-3-pro-preview",
            api_base="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        with dspy.context(lm=lm_gemini, cache=False):
            visual_analyzer = VisualAnalyzerWithTools(image_bytes=image_bytes)
            res = visual_analyzer()
        return res


def parse_args():
    parser = argparse.ArgumentParser(description="Classify and process a single file.")
    parser.add_argument("file", help="Path to the file to process")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    from dotenv import load_dotenv

    load_dotenv("../.env", override=True)
    # Load LLM config
    LM_CONFIG = {
        "model": os.getenv("DSPY_MODEL"),
        "api_key": os.getenv("DSPY_API_KEY"),
        "api_base": os.getenv("DSPY_ENDPOINT"),
        "api_version": os.getenv("DSPY_API_VERSION"),
        "max_tokens": int(os.getenv("DSPY_MAX_TOKENS", 50_000)),
        "temperature": float(os.getenv("DSPY_TEMPERATURE", 1.0)),
    }

    LM_CONFIG_SMALL = {
        "model": os.getenv("DSPY_FAST_MODEL"),
        "api_key": os.getenv("DSPY_API_KEY"),
        "api_base": os.getenv("DSPY_ENDPOINT"),
        "api_version": os.getenv("DSPY_FAST_API_VERSION"),
        "max_tokens": int(os.getenv("DSPY_FAST_MAX_TOKENS", 50_000)),
    }

    LM_CONFIG_IMAGE = {
        "model": os.getenv("DSPY_IMAGE_MODEL"),
        "api_key": os.getenv("DSPY_IMAGE_API_KEY"),
        "api_base": os.getenv("DSPY_IMAGE_ENDPOINT"),
    }

    lm = dspy.LM(**LM_CONFIG)
    lm_small = dspy.LM(**LM_CONFIG_SMALL)
    lm_image = dspy.LM(**LM_CONFIG_IMAGE)
    # dspy.configure(lm=lm_small)
    dspy.configure(lm=lm_image)

    result = asyncio.run(process_single_file(args.file))
    try:
        pprint(dict(result))
    except Exception:
        print(result)
