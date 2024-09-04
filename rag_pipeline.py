import torch
from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.utils import ComponentDevice
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

def run():
    device = "cuda:0" if torch.cuda.is_available() else "cpu" #for text embedder

    #Initializing each component of the Pipeline
    #documents will be retrived from ChromaDB
    document_store = ChromaDocumentStore(persist_path = "ChromaDB")
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-mpnet-base-v2", device = ComponentDevice(device))

    prompt_template = get_prompt()

    generator = LlamaCppGenerator(
        model="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 
        n_ctx=1024, #512 normally
        n_batch=128,
        model_kwargs={"n_gpu_layers": -1},
        generation_kwargs={"max_tokens": 256, "temperature": 0.1}, #change to 512 or 256
    )


    generator.warm_up()
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

    rag_pipeline.draw("diagrams/rag_pipeline.png") #generates a diagram of the pipeline in a png

    comfirmation_question = "Based on the Sample Guideline did the syllabus sample follow all the requirements?"
    result = rag_pipeline.run(
        data={
            "prompt_builder": {"query": comfirmation_question},
            "text_embedder": {"text": comfirmation_question},
            "answer_builder": {"query" : comfirmation_question},}
        )
    
    print(f"Document Store contains: {document_store.count_documents()} documents.")
    generated_answer = result['answer_builder']['answers'][0]
    print(generated_answer.data)

def get_prompt():
    prompt = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

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
