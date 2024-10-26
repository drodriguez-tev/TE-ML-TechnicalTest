from haystack import Pipeline
from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder, HuggingFaceAPITextEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.components.converters import PyPDFToDocument

from haystack.utils import Secret

from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever

class RAG:
    """
    Class that represents the RAG that will be used by the API.

    ...

    Attributes
    ----------
    huggingface_token : str
        Huggingface Hub token for allowing access to its models.
    document_store : MilvusDocumentStore
        Haystack representation of the Milvus Document Store.
    rag_pipeline : Pipeline
        Haystack pipeline for answering the user's question based on the information retrieved from the document store.
    indexing_pipeline : Pipeline()
        Haystack pipeline for adding the embedded document to the document store.
    milvus_port : int
        The port for the Milvus docker container connection.

    Methods
    -------
    setup(file_path):
        Sets up the required pipelines so the RAG can be ready to recieve and answer questions.
    initialize_document_store():
        Initializes the Haystack Milvus document store for storing PDF documents.
    create_indexing_pipeline():
        Creates the indexing pipeline for converting, embedding, and storing the PDF document in the Milvus document store.
    run_indexing(file_path):
        Runs the indexing pipeline for processing and storing the PDF document in 'file_path'.
    create_rag_pipeline():
        Creates the rag pipeline for embedding the user's question, retrieving information from the PDF that is relevant to the question,
        and generates a response based on this information.
    run_rag(question):
        Runs the RAG pipeline. Receives a 'question' and returns an appropriate answer based on the retrieved information from the PDF.
    """

    CONFIG_PROMPT ="""Answer the following query based on the provided context. If the context does
                        not include an answer, reply with 'I don't know'.\n
                        Query: {{query}}
                        Documents:
                        {% for doc in documents %}
                            {{ doc.content }}
                        {% endfor %}
                        Answer:
                    """       

    def __init__(self, huggingface_token, milvus_port):
        """
        Constructs the attributes for the RAG class.
    
        Parameters
        ----------
            huggingface_token : str
                Huggingface Hub token for allowing access to its models.
            milvus_port : int
                The port for the Milvus docker container connection.
        """

        self.milvus_port = milvus_port
        self.document_store = None
        self.rag_pipeline = Pipeline()
        self.indexing_pipeline = Pipeline()
        self.huggingface_token = huggingface_token
            
    def setup(self, file_path):
        """
        Sets up the required pipelines for the RAG to use to recieve and answer questions based on the PDF file in 'file_path'.
        
        Parameters
        ----------
            file_path : str
                The file path of the PDF document from which information is to be extracted.
        
        Returns
        -------
        None
        """

        self.initialize_document_store()
        self.create_indexing_pipeline()
        self.run_indexing(file_path)
        self.create_rag_pipeline()
    
    def initialize_document_store(self):
        """
        Initializes the Haystack Milvus document store for storing PDF documents. Uses the provided 'milvus_port' in the
        class initialization.
        
        Parameters
        ----------
        NONE
        
        Returns
        -------
        NONE
        """

        self.document_store = MilvusDocumentStore(
            #connection_args={"uri": f"http://localhost:{self.milvus_port}"},
            connection_args={"uri": f"http://standalone:{self.milvus_port}"},
            drop_old=True,
        )
            
    def create_indexing_pipeline(self):
        """
        Creates the indexing pipeline for converting, embedding, and storing the PDF document in the Milvus document store.
        
        Parameters
        ----------
        NONE
        
        Returns
        -------
        NONE
        """

        # Define components
        splitter = DocumentSplitter(split_by="sentence", split_length=2)
        document_embedder = HuggingFaceAPIDocumentEmbedder(api_type="serverless_inference_api",
                                                api_params={"model": "BAAI/bge-small-en-v1.5"},
                                                token=Secret.from_token(self.huggingface_token))
        writer = DocumentWriter(self.document_store)

        # Create Pipeline
        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("converter", PyPDFToDocument())
        self.indexing_pipeline.add_component("splitter", splitter)
        self.indexing_pipeline.add_component("embedder", document_embedder)
        self.indexing_pipeline.add_component("writer", writer)
        self.indexing_pipeline.connect("converter", "splitter")
        self.indexing_pipeline.connect("splitter", "embedder")
        self.indexing_pipeline.connect("embedder", "writer")
    
    def run_indexing(self, file_path):
        """
        Runs the indexing pipeline for processing and storing the PDF document in 'file_path'.
        
        Parameters
        ----------
            file_path : str
                File path of the PDF file from which the information will be retrieved.
        
        Returns
        -------
        NONE
        """

        self.indexing_pipeline.run({"converter": {"sources": [file_path]}})
        
    def create_rag_pipeline(self):
        """
        Creates the rag pipeline for embedding the user's question, retrieving information from the PDF that is relevant to the question,
        and generates a response based on this information.
        
        Parameters
        ----------
        NONE
        
        Returns
        -------
        NONE
        """

        text_embedder = HuggingFaceAPITextEmbedder(api_type="serverless_inference_api", api_params={"model": "BAAI/bge-small-en-v1.5"}, token=Secret.from_token(self.huggingface_token))
        generator = HuggingFaceAPIGenerator(api_type="serverless_inference_api", api_params={"model": "HuggingFaceH4/zephyr-7b-beta"}, token=Secret.from_token(self.huggingface_token))
        self.rag_pipeline.add_component("text_embedder", text_embedder)
        self.rag_pipeline.add_component("retriever", MilvusEmbeddingRetriever(document_store=self.document_store, top_k=3))
        self.rag_pipeline.add_component("prompt_builder", PromptBuilder(template=RAG.CONFIG_PROMPT))
        self.rag_pipeline.add_component("generator", generator)
        self.rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.rag_pipeline.connect("prompt_builder", "generator")

    def run_rag(self, question):
        """
        Runs the RAG pipeline. Receives a 'question' and returns an appropriate answer based on the retrieved information from the PDF.
        
        Parameters
        ----------
            question : str
                User's question that will be answered based on the stored PDF file.
        
        Returns
        -------
            answer : str
                The generated answer to the user's question.
        """

        results = self.rag_pipeline.run(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {"query": question},
            }
        )
        answer = results["generator"]["replies"][0]
        return answer

