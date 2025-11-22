from typing import Literal
import dspy
from attachments import set_verbose
from attachments.dspy import Attachments

set_verbose(False)

DOCUMENT_TYPES = [
    "SEC_FILING",
    "PATENT_FILING",
    "CONTRACT",
    "CITY_INFRASTRUCTURE",
    "OTHER",
]

SEC_FILING_TYPES = [
    "10-K",
    "10-Q",
    "8-K",
    "DEF 14A",
    "DEF 14B",
    "DEF 14C",
    "FORM 4",
    "OTHER",
]


class SECDocumentType(dspy.Signature):
    """Classify a SEC filing into a type."""

    document_images: dspy.Image = dspy.InputField()
    document_type: Literal[tuple(SEC_FILING_TYPES)] = dspy.OutputField(
        desc="One of the following: " + ", ".join(SEC_FILING_TYPES)
    )


class DocumentType(dspy.Signature):
    """Classify a document into a type based on first few page images."""

    document_images: list[dspy.Image] = dspy.InputField()
    document_type: Literal[tuple(DOCUMENT_TYPES)] = dspy.OutputField(
        desc="One of the following: " + ", ".join(DOCUMENT_TYPES)
    )


class DocumentClassifier(dspy.Module):
    """Classify a document into a type."""

    def __init__(self, document_path: str):
        super().__init__()
        self.classifier = dspy.Predict(DocumentType)

        document_images = [
            dspy.Image(img)
            for img in self._extract_pdf_images(document_path, max_pages=2)
        ]
        self.document_images = document_images

    def _extract_pdf_images(
        self, document_path: str, max_pages: int = 3
    ) -> list[dspy.Image]:
        imgs = Attachments(document_path + "[tile:1x1]").images[:max_pages]
        return imgs

    def forward(self) -> str:
        return self.classifier(document_images=self.document_images)

    async def aforward(self) -> str:
        return await self.classifier.acall(document_images=self.document_images)
