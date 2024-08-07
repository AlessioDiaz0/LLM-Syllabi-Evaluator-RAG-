from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.utils import ComponentDevice
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever


from utils import read_input_json, serialize_generated_answer

def run():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #Initializing each component of the Pipeline
    document_store = ChromaDocumentStore(persist_path = persist_path)
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

    prompt_template = """
    Given the following information, answer the question.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    generator = LlamaCppGenerator(
        model="/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M-take2.gguf", 
        n_ctx=512,
        n_batch=128,
        model_kwargs={"n_gpu_layers": -1},
        generation_kwargs={"max_tokens": 128, "temperature": 0.1},
    )

    #check embedder later
    text_embedder = SentenceTransformersTextEmbedder(
        model=f"sentence-transformers/distiluse-base-multilingual-cased-v1", device=ComponentDevice(device)
    )

    generator.warm_up()
    text_embedder.warm_up()


    llm = generator()

    rag_pipeline = Pipeline()
    #rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("embedder_retriever", ChromaEmbeddingRetriever(document_store=document_store))
    rag_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    rag_pipeline.add_component("answer_builder", AnswerBuilder())
    rag_pipeline.add_component("llm", generator)

    # rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    # rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    # rag_pipeline.connect("prompt_builder", "llm")

    rag_pipeline.draw("diagrams/rag_pipeline.png") #generates a diagram of the pipeline in a png

    result = rag_pipeline.run(data={"prompt_builder": {"query":query}, "text_embedder": {"text": query}})
    print(result["llm"]["replies"][0])



if __name__ == "__main__":
    run()



other_prompt = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        Answer the query from the context provided.
        If it is possible to answer the question from the context, copy the answer from the context.
        If the answer in the context isn't a complete sentence, make it one.

        Context:
        {% for doc in documents %}
        {{ doc.content }}
        {% endfor %};
        <eot_id><start_header_id|>user<|end_header_id|>
        query: {{query}}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        Answer:
        """