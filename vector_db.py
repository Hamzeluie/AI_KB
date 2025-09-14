import json
import uuid
from time import time
from typing import Dict, List, Optional

import yaml

# from kb.hf_emb_model import EmbeddingModel
from hf_emb_model import EmbeddingModel
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)
from qdrant_client import QdrantClient, models


class MultiTenantVectorDB:
    # ... (The __init__, _load_config, _ensure_collection_exists, and _split_documents methods remain unchanged) ...
    """
    A class to manage a multi-tenant Qdrant vector database, handling
    document ingestion, searching, and data management with user isolation.
    It can dynamically split text or JSON documents.
    """

    def __init__(
        self,
        config_path: str = "emb.yaml",
        collection_name: str = "multitenant_rag_collection",
    ):
        self.config = self._load_config(config_path)
        self.client = QdrantClient(path=self.config["vectorstore"]["persist_dir"])
        self.embedding_model = EmbeddingModel(
            model_name=self.config["embedding"]["model"], device="cuda"
        )
        self.collection_name = collection_name
        self._ensure_collection_exists()

    def _load_config(self, config_path: str) -> Dict:
        """Loads configuration from a YAML file."""
        try:
            with open(config_path, "r") as file:
                print(f"Configuration loaded from {config_path}")
                return yaml.safe_load(file)
        except Exception as e:
            raise IOError(f"Error loading configuration from {config_path}: {e}")

    def _ensure_collection_exists(self):
        """Ensures the collection and necessary payload indexes exist."""
        # ... (This method remains unchanged from the previous version)
        collections_response = self.client.get_collections()
        existing_collections = [c.name for c in collections_response.collections]

        if self.collection_name in existing_collections:
            return

        print(f"Collection '{self.collection_name}' not found. Creating...")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.config["embedding"]["dimension"],
                distance=models.Distance.COSINE,
            ),
        )
        print("Creating payload indexes for 'owner_id' and 'kb_id'...")
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="owner_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="kb_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print(f"Collection '{self.collection_name}' and indexes created successfully.")

    def _split_documents(
        self, documents: List[Document], doc_type: str
    ) -> List[Document]:
        """
        Selects the appropriate splitter based on doc_type and splits the documents.
        """
        if doc_type == "text":
            print("Using RecursiveCharacterTextSplitter for plain text.")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config["document_processing"]["chunk_size"],
                chunk_overlap=self.config["document_processing"]["chunk_overlap"],
            )
            doc_document = [
                Document(
                    page_content=documents.page_content, metadata=documents.metadata
                )
            ]
            return text_splitter.split_documents(doc_document)

        elif doc_type == "json":
            print("Attempting to use RecursiveJsonSplitter.")
            if not documents:
                return []

            # Assume the full JSON is in the first document's content
            full_json_content = documents.page_content
            source_metadata = documents.metadata

            try:
                json_data = json.loads(full_json_content)
                json_splitter = RecursiveJsonSplitter(
                    max_chunk_size=self.config["document_processing"]["chunk_size"],
                    min_chunk_size=self.config["document_processing"]["chunk_overlap"],
                )
                # This returns a list of dictionaries (the chunks)
                dict_chunks = json_splitter.split_json(json_data=json_data)

                # Convert the dictionary chunks back into Document objects
                json_doc_chunks = []
                for i, chunk in enumerate(dict_chunks):
                    chunk_metadata = source_metadata.copy()
                    chunk_metadata["chunk_id"] = i
                    # @Borhan: English
                    # doc = Document(page_content=json.dumps(chunk), metadata=chunk_metadata)
                    # @Borhan: Arabic
                    doc = Document(
                        page_content=json.dumps(chunk, ensure_ascii=False),
                        metadata=chunk_metadata,
                    )
                    json_doc_chunks.append(doc)

                print(f"Successfully split JSON into {len(json_doc_chunks)} chunks.")
                return json_doc_chunks

            except Exception as e:
                print(
                    f"RecursiveJsonSplitter failed: {e}. Falling back to character splitting the raw JSON."
                )
                # Fallback: Treat the raw JSON string as plain text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config["document_processing"]["chunk_size"],
                    chunk_overlap=self.config["document_processing"]["chunk_overlap"],
                )
                return text_splitter.split_documents(documents)
        else:
            raise ValueError(
                f"Unsupported doc_type: '{doc_type}'. Must be 'text' or 'json'."
            )

    def kb_exists(self, owner_id: str, kb_id: str) -> bool:
        """
        Checks if a knowledge base already exists for a given user.

        Returns:
            bool: True if any points exist for the owner_id/kb_id pair, False otherwise.
        """
        response = self.client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="owner_id", match=models.MatchValue(value=owner_id)
                    ),
                    models.FieldCondition(
                        key="kb_id", match=models.MatchValue(value=kb_id)
                    ),
                ]
            ),
            exact=True,
        )
        return response.count > 0

    def add_documents(
        self,
        owner_id: str,
        kb_id: str,
        documents: List[Document],
        doc_type: str,
        overwrite: bool = False,
    ):
        """
        Adds or updates a knowledge base. If the kb_id already exists for the user,
        it can either be skipped or overwritten.

        Args:
            overwrite (bool): If True, will delete existing documents for this kb_id and add new ones.
                              If False, will skip the upload if the kb_id already exists.
        """
        print(
            f"\nProcessing request to add documents for user='{owner_id}', kb='{kb_id}'..."
        )

        # --- NEW: Check for existence ---
        if self.kb_exists(owner_id, kb_id):
            if not overwrite:
                print(
                    f"Knowledge base '{kb_id}' already exists for user '{owner_id}'. Skipping upload."
                )
                print("To update, set overwrite=True.")
                return
            else:
                print(
                    f"Knowledge base '{kb_id}' exists. Deleting old version before adding new one."
                )
                self.delete_documents(owner_id, kb_id)

        # --- The rest of the logic remains the same ---
        print(f"Input JSON content: {documents.page_content}")
        chunks = self._split_documents(documents, doc_type)
        print(f"Total chunks to process: {len(chunks)}.")

        points_to_upsert = []
        for chunk in chunks:
            point_id = str(uuid.uuid4())
            embedding = self.embedding_model.embed_query(chunk.page_content)
            points_to_upsert.append(
                models.PointStruct(
                    id=point_id,
                    vector=self.embedding_model.to_list(embedding),
                    payload={
                        "owner_id": owner_id,
                        "kb_id": kb_id,
                        "page_content": chunk.page_content,
                        "metadata": chunk.metadata,
                    },
                )
            )

        if not points_to_upsert:
            print("No chunks to add.")
            return

        self.client.upsert(
            collection_name=self.collection_name, points=points_to_upsert, wait=True
        )
        print(f"Successfully added {len(points_to_upsert)} points to the database.")

    def search(
        self, owner_id: str, query_text: str, kb_id: List[str] = None, limit: int = 5
    ) -> List[Dict]:
        """
        Performs a filtered vector search for a specific user and optionally a specific knowledge base.
        """
        print(
            f"\n--- Searching for user='{owner_id}'"
            + (f" in kb='{kb_id}'" if kb_id else "")
            + " ---"
        )
        query_embedding = self.embedding_model.embed_query(query_text)
        filter_conditions = [
            models.FieldCondition(
                key="owner_id", match=models.MatchValue(value=owner_id)
            )
        ]
        if kb_id:
            filter_conditions.append(
                models.FieldCondition(key="kb_id", match=models.MatchAny(any=kb_id))
            )
        query_filter = models.Filter(must=filter_conditions)

        search_response = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embedding_model.to_list(query_embedding),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        points_list = search_response.points

        results = [
            {
                "score": hit.score,
                "content": hit.payload.get("page_content"),
                "payload": hit.payload,
            }
            for hit in points_list
        ]
        print(f"Found {len(results)} results for query: '{query_text}'")
        for res in results:
            # print(f"  Score: {res['score']:.4f} | Content: '{res['content'][:100]}...'")
            print(f"  Score: {res['score']:.4f} | Content: '{res['content']}...'")
        return results

    def delete_documents(self, owner_id: str, kb_id: str):
        """
        Deletes all points associated with a specific user's knowledge base.
        """
        print(
            f"\n--- Deleting documents for owner_id='{owner_id}' and kb_id='{kb_id}' ---"
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="owner_id", match=models.MatchValue(value=owner_id)
                        ),
                        models.FieldCondition(
                            key="kb_id", match=models.MatchValue(value=kb_id)
                        ),
                    ]
                )
            ),
            wait=True,
        )
        print("Deletion complete.")

    def count_points(self) -> int:
        """Returns the total number of points in the collection."""
        count_result = self.client.count(
            collection_name=self.collection_name, exact=True
        )
        return count_result.count

    def see_payload_for_owner_kb(self, owner_id: str, kb_id: str):
        # Option 1: Scroll through all records (for large collections)
        records = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="owner_id", match=models.MatchValue(value=owner_id)
                    ),
                    models.FieldCondition(
                        key="kb_id", match=models.MatchValue(value=kb_id)
                    ),
                ]
            ),
            limit=100,  # Adjust batch size
            with_payload=True,  # Include payloads
            with_vectors=False,  # Exclude vectors to reduce payload size
        )
        print(records)
        return records
        # for record in records:
        #     print(f"ID: {record.id}, Payload: {record.payload}")

    def see_payload(self):
        # Option 1: Scroll through all records (for large collections)
        records = self.client.scroll(
            collection_name=self.collection_name,
            limit=100,  # Adjust batch size
            with_payload=True,  # Include payloads
            with_vectors=False,  # Exclude vectors to reduce payload size
        )
        print(records)
        return records
        # for record in records:
        #     print(f"ID: {record.id}, Payload: {record.payload}")


if __name__ == "__main__":
    # Example usage
    db = MultiTenantVectorDB(config_path="emb.yaml")
    db.see_payload()
