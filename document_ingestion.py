import os
import torch

from haystack.components.converters import (
    MarkdownToDocument,
    PyPDFToDocument,
    TextFileToDocument,
)
from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.utils import ComponentDevice
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.routers import FileTypeRouter
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.joiners.document_joiner import DocumentJoiner

def run():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    #There is also an option to fetch the guideline/requirements data given an URL
    source_guideline_folder = list_files_in_folder("documents/syllabus_guideline")
    source_input_folder = list_files_in_folder("documents/syllabus_input")

    #
    def create_document_pipeline(persist_path):
        #Initializing each component of the Pipeline
        document_store = ChromaDocumentStore(persist_path = persist_path)
        document_pipeline = Pipeline()
        document_pipeline.add_component(
            "document_embedder",
            SentenceTransformersDocumentEmbedder(
                model = 'sentence-transformers/all-mpnet-base-v2',
                device = ComponentDevice(device), #ensures embedder runs on gpu if available # type: ignore
                batch_size = 32,
                normalize_embeddings = False
            )
        )

        document_pipeline.add_component(
            "document_writer",
            DocumentWriter(document_store = document_store, policy=DuplicatePolicy.OVERWRITE)
        )

        #Files are routed based on their type to different outputs 
        #There is the possibility of adding images
        document_pipeline.add_component(
            "file_type_router",
            FileTypeRouter(mime_types=["application/pdf", "text/markdown", "text/plain"]),
        )

        document_pipeline.add_component("pdf_converter", PyPDFToDocument())
        document_pipeline.add_component("markdown_converter", MarkdownToDocument())
        document_pipeline.add_component("text_file_converter", TextFileToDocument())


        #Possibility of clean out regex patterns "remove_regex: Optional[str] = None"
        document_pipeline.add_component("document_cleaner", DocumentCleaner(remove_regex = None))

        #Document splitter performs best with default settings
        document_pipeline.add_component(
            "document_splitter",
            DocumentSplitter())

        #join_mode has also the possibility of concatenation from multiple documents and discard any duplicates
        document_pipeline.add_component(
            "document_joiner",
            DocumentJoiner(join_mode = "merge"))


        document_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
        document_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
        document_pipeline.connect("file_type_router.application/pdf", "pdf_converter.sources")
        document_pipeline.connect("text_file_converter", "document_joiner")
        document_pipeline.connect("markdown_converter", "document_joiner")
        document_pipeline.connect("pdf_converter", "document_joiner")
        document_pipeline.connect("document_joiner", "document_cleaner")
        document_pipeline.connect("document_cleaner", "document_splitter")
        document_pipeline.connect("document_splitter", "document_embedder")
        document_pipeline.connect("document_embedder", "document_writer")

        #generates a diagram of the pipeline in a png
        #document_pipeline.draw("diagrams/document_ingestion.png") 

        return document_store, document_pipeline

    #guideline_vector
    doc_count_1, document_guideline_pipeline = create_document_pipeline("ChromaDB")
    document_guideline_pipeline.run({"file_type_router": {"sources": source_guideline_folder}})
    print("Processed: ", doc_count_1.count_documents(), "file(s)")

    #input_vector
    doc_count_2, document_input_pipeline = create_document_pipeline("ChromaDB")
    document_input_pipeline.run({"file_type_router": {"sources": source_input_folder}})
    print("Processed: ", doc_count_2.count_documents(), "file(s)")


def list_files_in_folder(folder_path):
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

if __name__ == "__main__":
    run()