import dspy
import pymupdf4llm


class ProduceGist(dspy.Signature):
    """
    Produce 3 - 5 sentence gist of what this chunk is about, so we can assign it to a heading.
    """

    toc_path: list[str] = dspy.InputField(
        desc="path down which this chunk has traveled so far in the Table of Contents"
    )
    chunk: str = dspy.InputField()
    gist: str = dspy.OutputField()


class ProduceHeaders(dspy.Signature):
    """
    Produce a list of headers (top-level Table of Contents) for structuring a report on *all* chunk contents.
    Make sure every chunk would belong to exactly one section.
    Ultimately your task is to answer the question: "What are the top reasons of rejections and corresponding recommendations by FDA?"
    """

    toc_path: list[str] = dspy.InputField()
    chunk_summaries: str = dspy.InputField()
    headers: list[str] = dspy.OutputField()


class WriteSection(dspy.Signature):
    """
    Craft a Markdown section, given a path down the table of contents, which ends with this section's specific heading.
    Start the content right beneath that heading: use sub-headings of depth at least +1 relative to the ToC path.
    Your section's content is to be entirely derived from the given list of chunks. That content must be complete but
    very concise, with all necessary knowledge from the chunks reproduced and repetitions or irrelevant details omitted. If a certain heading does not have any content, just omit it.
    """

    toc_path: list[str] = dspy.InputField()
    content_chunks = dspy.InputField()
    section_content = dspy.OutputField()


def massively_summarize(
    toc_path: list[str],
    chunks: list[str],
):
    if len(chunks) < 5 or len(toc_path) >= 3:
        content = dspy.ChainOfThought(WriteSection)(
            toc_path=toc_path, content_chunks=chunks
        ).section_content
        return f"{toc_path[-1]}\n\n{content}"

    # Produce gist for each chunk
    gister = dspy.ChainOfThought(ProduceGist)
    chunk_summaries = gister.batch(
        [
            dspy.Example(toc_path=toc_path, chunk=chunk).with_inputs(
                "toc_path", "chunk"
            )
            for chunk in chunks
        ]
    )
    chunk_summaries = [summary.gist for summary in chunk_summaries]

    # Get headers for all summary chunks
    produce_headers = dspy.ChainOfThought(ProduceHeaders)
    headers = produce_headers(
        toc_path=toc_path, chunk_summaries=chunk_summaries
    ).headers

    # Classify each chunk under a header
    classifier = dspy.ChainOfThought(
        f"toc_path: list[str], chunk -> topic: Literal{headers}"
    )
    topics = classifier.batch(
        [
            dspy.Example(toc_path=toc_path, chunk=chunk).with_inputs(
                "toc_path", "chunk"
            )
            for chunk in chunks
        ]
    )

    sections = {topic: [] for topic in headers}
    for topic, chunk in zip(topics, chunks):
        sections[topic.topic].append(chunk)

    # Recursively summarize each section
    parallel_massively_summarize = massively_summarize
    summarized_sections = [
        parallel_massively_summarize(toc_path=toc_path + [topic], chunks=section_chunks)
        for topic, section_chunks in sections.items()
    ]

    return toc_path[-1] + "\n\n" + "\n\n".join(summarized_sections)


class Summarizer(dspy.Module):
    """Summarize a document of arbitrary length."""

    def __init__(self, document_path: str):
        super().__init__()
        self.document_path = document_path
        self.document_chunks = pymupdf4llm.to_markdown(self.document_path)

    def forward(self) -> str:
        chunks_to_process = self.document_chunks.split("##")
        return massively_summarize(toc_path=["# Key Themes"], chunks=chunks_to_process)

    async def aforward(self) -> str:
        return self.forward()
