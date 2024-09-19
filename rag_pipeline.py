import torch
from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.utils import ComponentDevice
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

from utils import save_to_json

def run():
    device = "cuda:0" if torch.cuda.is_available() else "cpu" #for text embedder

    # Initializing each component of the Pipeline
    # documents will be retrived from ChromaDB
    document_store = ChromaDocumentStore(persist_path = "ChromaDB")
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-mpnet-base-v2", device = ComponentDevice(device))

    prompt_template = get_prompt()

    #more information on parameterization can be found on
    #https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
    generator = OllamaGenerator(
        model="llama3.1",
        url="http://localhost:11434/api/generate",
        generation_kwargs={"temperature": 0.1, "top_k": 20, "top_p": 0.5},
        timeout=300 #timeout increased for slower devices
    )

    text_embedder.warm_up()

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("embedder_retriever", ChromaEmbeddingRetriever(document_store=document_store))
    rag_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    rag_pipeline.add_component("answer_builder", AnswerBuilder())
    rag_pipeline.add_component("llm", generator)

    rag_pipeline.connect("text_embedder", "embedder_retriever")
    rag_pipeline.connect("embedder_retriever.documents", "prompt_builder.documents")  # Enforce passing documents
    rag_pipeline.connect("embedder_retriever", "prompt_builder")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("llm.meta", "answer_builder.meta")
    rag_pipeline.connect("embedder_retriever", "answer_builder.documents")
    rag_pipeline.connect("embedder_retriever.documents", "answer_builder.documents")

    #rag_pipeline.draw("diagrams/rag_pipeline.png") #generates a diagram of the pipeline in a png

    #Based on the Sample Guideline did the syllabus sample follow all the requirements
    #Give feedback on the sample_syllabus using the two category lists
    comfirmation_question = (
        "List everything in two categories: important and minor. "
        "Important: anything that is missing from the guidelines or requirements. "
        "Minor: anything additional that will help the students in the course.\n"
        "Provide between 0 to 5 bullet points per category.\n"
        "Format the output as follows:\n"
        "Important:\n- [item 1]\n- [item 2]\nMinor:\n- [item 1]\n"
        "Output examples:\n"
        "Perfect syllabus:\nImportant:\n- None\nMinor:\n- None\n"
        "Other syllabus:\nImportant:\n- Office hours are missing\n- Textbook is missing\n"
        "Minor:\n- Could add links to school resources\n"
        "Wrong outputs will use: could be more detailed or specific"
    )
    result = rag_pipeline.run(
        data={
            "prompt_builder": {"query": comfirmation_question},
            "text_embedder": {"text": comfirmation_question},
            "answer_builder": {"query" : comfirmation_question},}
        )
    
    print(f"Document Store contains: {document_store.count_documents()} documents.")
    save_to_json(result)

def get_prompt():
    prompt = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        Task: Review the syllabus and only list missing elements from the guidelines or requirements.
        Do not include anything that is correct or well-explained.
        Be concise and only list negatives.
        
        Context:
        {% for doc in documents %}
        {{ doc.content }}
        {% endfor %};
        <eot_id><start_header_id|>user<|end_header_id|>
        query: {{query}}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        Answer:
        """
    return prompt


if __name__ == "__main__":
    run()
