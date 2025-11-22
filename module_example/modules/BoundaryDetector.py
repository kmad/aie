import asyncio
import logging
from typing import Dict, List, Literal, Tuple

import dspy
from dspy import asyncify
from attachments.dspy import Attachments

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page classification classes
CLASSES = (
    "COVER PAGE",
    "TERMS_AND_CONDITIONS",
    "SIGNATURE_PAGE",
    "START_OF_APPENDIX",
    "START_OF_EXHIBIT",
    "START_OF_SCHEDULE",
)


class ClassifyPage(dspy.Signature):
    """Classifies a single page from a PDF order form into one of several predefined classes."""

    page_image: dspy.Image = dspy.InputField(
        desc="The image of a single page from the PDF file."
    )
    page_class = dspy.OutputField(desc="The type or class of the page.")


class PageClassifier(dspy.Module):
    """DSPy module for classifying PDF pages into predefined categories."""

    def __init__(self):
        super().__init__()
        classify_signature = ClassifyPage.with_updated_fields(
            "page_class",
            type_=Literal[tuple(CLASSES)],  # type: ignore
        )
        self.predictor = dspy.Predict(classify_signature)

    def forward(self, page_image: dspy.Image):
        """Classify a single page image."""
        return self.predictor(page_image=page_image)


class PDFBoundaryDetector:
    """
    A class to detect boundaries in PDF documents by converting pages to images
    and classifying them to identify document sections.
    """

    def __init__(self, pdf_file: str):
        self.pdf_file = pdf_file
        self.page_images: List[dspy.Image] = []
        self.page_classifications: Dict[int, str] = {}
        self.document_boundaries: Dict[str, Tuple[int, int]] = {}
        self.page_classifier = asyncify(PageClassifier())
        self.boundary_detector = None
        self.page_images = [
            dspy.Image(img) for img in Attachments(self.pdf_file + "[tile:1x1]").images
        ]

    def get_page_images(self, pages: list[int]) -> List[dspy.Image]:
        """Get the page images."""
        return [self.page_images[i] for i in pages]

    async def classify_page(self, i: int, img: dspy.Image) -> Tuple[int, str]:
        result = await self.page_classifier(page_image=img)
        return i, result.page_class

    def get_page_classifications(self) -> Dict[int, str]:
        """Get the page classifications."""
        return self.page_classifications

    async def aforward(self) -> Dict[str, Tuple[int, int]]:
        # Classify all pages concurrently

        tasks = [self.classify_page(i, img) for i, img in enumerate(self.page_images)]
        results = await asyncio.gather(*tasks)

        # Store classifications as list of tuples (page_num, classification)
        self.page_classifications = dict(results)
        logger.info(f"Page classifications: {self.page_classifications}")

        boundary_signature = dspy.Signature(
            "pages_and_classifications -> document_boundaries: dict[str, tuple[int, int]]"
        ).with_instructions(
            "Detect boundaries within a document. For example, in a contract, you should separate the main body from the appendices and exhibits. Each exhibit in its totality should be a separate document. Name each section consistently"
        )
        self.detector = dspy.ReAct(
            boundary_signature, tools=[self.get_page_images], max_iters=10
        )

        response = self.detector(pages_and_classifications=self.page_classifications)

        logger.info(f"Boundary detection response: {response}")
        self.document_boundaries = response.document_boundaries
        return self.document_boundaries
