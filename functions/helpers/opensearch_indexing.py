from helpers.rag_helpers import create_opensearch_client, create_index_if_not_exists, index_document_bulk

def opensearch_indexing(embeddings, chunks, tenant_id, document_type, object_key, filename):

    try:
        opensearch_client = create_opensearch_client()
        
        index_name = f"rag-documents-{tenant_id}"
        
        # Crear índice si no existe
        index_created = create_index_if_not_exists(
            opensearch_client, 
            index_name,     
            dimensions=1024
        )
        
        if not index_created:
            print(f"⚠️ No se pudo crear/verificar índice {index_name}")
            return {
                "success": False,
                "message": f"Error creando índice {index_name}"
            }
        
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            documents.append({
                'content': chunk,
                'embedding': embedding,
                'document_type': document_type,
                'file_format': '.pdf',
                'source_file': object_key
            })
        
        indexing_success = index_document_bulk(
            opensearch_client,
            index_name,
            documents,
            tenant_id
        )
        
        if indexing_success:
            print(f"🎉 Documento indexado exitosamente en OpenSearch")
            return {
                "success": True,
                "message": f"PDF procesado e indexado: {len(chunks)} chunks",
                "details": {
                    "tenant_id": tenant_id,
                    "index_name": index_name,
                    "chunks_count": len(chunks),
                    "embeddings_count": len(embeddings),
                    "document_type": document_type,
                    "filename": filename
                }
            }
        else:
            return {
                "success": False,
                "message": "Error en indexado bulk de OpenSearch"
            }
                
    except Exception as opensearch_error:
        print(f"❌ Error en OpenSearch: {str(opensearch_error)}")
        return {
            "success": False,
            "message": f"Error en OpenSearch: {str(opensearch_error)}"
        }


def opensearch_query(question_embedding, tenant_id, document_type=None):

    try:
        from helpers.rag_helpers import create_opensearch_client
        
        opensearch_client = create_opensearch_client()
        
        index_name = f"rag-documents-{tenant_id}"
        
        if not opensearch_client.indices.exists(index=index_name):
            return {
                "success": True,
                "documents": [],
                "total_found": 0,
                "message": f"No hay documentos indexados para el tenant {tenant_id}"
            }
        
        search_query = {
            "size": 10,  # Top 10 documentos más relevantes
            "query": {
                "bool": {
                    "must": {
                        "knn": {
                            "embedding": {
                                "vector": question_embedding,
                                "k": 10
                            }
                        }
                    },
                    "filter": [
                        {"term": {"tenant_id": tenant_id}}
                    ]
                }
            },
            "_source": [
                "content", 
                "source_file", 
                "document_type", 
                "chunk_index", 
                "created_at",
                "document_hash"
            ]
        }
        
        if document_type:
            search_query["query"]["bool"]["filter"].append(
                {"term": {"document_type": document_type}}
            )
            print(f"📂 Filtrando por document_type: {document_type}")
        
        print(f"🔎 Ejecutando búsqueda en índice: {index_name}")
        
        response = opensearch_client.search(
            index=index_name,
            body=search_query
        )
        
        hits = response.get('hits', {})
        total_found = hits.get('total', {}).get('value', 0)
        documents = []
        
        for hit in hits.get('hits', []):
            source = hit.get('_source', {})
            score = hit.get('_score', 0)
            
            documents.append({
                'content': source.get('content', ''),
                'source_file': source.get('source_file', ''),
                'document_type': source.get('document_type', ''),
                'chunk_index': source.get('chunk_index', 0),
                'created_at': source.get('created_at', ''),
                'document_hash': source.get('document_hash', ''),
                'score': score
            })
        
        return {
            "success": True,
            "documents": documents,
            "total_found": total_found,
            "index_searched": index_name
        }
        
    except Exception as e:
        print(f"❌ Error en opensearch_query: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Error en búsqueda OpenSearch: {str(e)}",
            "documents": []
        }