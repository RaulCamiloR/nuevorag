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
        
        # Preparar documentos para indexado
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            documents.append({
                'content': chunk,
                'embedding': embedding,
                'document_type': document_type,
                'file_format': '.pdf',
                'source_file': object_key
            })
        
        # Indexar documentos en bulk
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