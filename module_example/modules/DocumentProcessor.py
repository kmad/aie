import dspy
from attachments.dspy import Attachments
from pydantic import BaseModel, Field


class DocumentProcessorModule(dspy.Module):
    """Document processor module with multimodal support."""

    def __init__(self):
        super().__init__()
        # self.classifier = dspy.ChainOfThought(DocumentClassifier)
        # self.customer_agreement_analyzer = dspy.ChainOfThought(
        #     CustomerAgreementAnalysis
        # )
        # self.visual_analyzer = dspy.ChainOfThought(VisualAnalysis)

    def _extract_pdf_images(self, pdf_path: str, max_pages: int = None):
        return Attachments(pdf_path).images[:max_pages]


class Transaction(BaseModel):
    """A transaction in the document."""

    transaction_type: str = Field(description="The type of transaction.")
    shares_sold: int = Field(description="The number of shares sold.")
    price_per_share: float = Field(description="The price per share.")
    total_price: float = Field(description="The total price of the transaction.")


class DocumentSchema(BaseModel):
    """A structured representation of the document content."""

    form_type: str = Field(description="The type of form filed.")
    filing_date: str = Field(
        description="The date of the filing in the format YYYY-MM-DD."
    )
    total_shares_sold: int
    transactions: list[Transaction] = Field(description="A list of transactions.")


class DocumentAnalyzerSchema(dspy.Signature):
    """Analyze document content and extract insights."""

    document: Attachments = dspy.InputField()
    document_schema: DocumentSchema = dspy.OutputField(
        desc="A structured representation of the document content."
    )


class Form4Processor(dspy.Module):
    """Process a FORM 4 document."""

    def __init__(self):
        super().__init__()

    def forward(self, doc: Attachments) -> DocumentSchema:
        result_schema = dspy.ChainOfThought(DocumentAnalyzerSchema)(document=doc)
        return result_schema.document_schema
