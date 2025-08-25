
import boto3
import json
import io
import os
import hashlib
import PyPDF2
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth



def extract_pdf_text(file_content: bytes) -> str:
    
    try:
        pdf_file = io.BytesIO(file_content)
        
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_content = ""

        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                page_text = page.extract_text()
                text_content += f"\n{page_text}\n"
            except Exception as e:
                print(f"Error en página {page_num}: {str(e)}")
                continue
        
        text_content = clean_extracted_text(text_content)
        
        return text_content
        
    except Exception as e:
        print(f"❌ Error extrayendo texto del PDF: {str(e)}")
        raise ValueError(f"No se pudo extraer texto del PDF: {str(e)}")


def clean_extracted_text(text: str) -> str:
    
    text = text.replace('\n\n\n', '\n\n')
    
    text = ' '.join(text.split())
    
    text = text.replace('\x00', '')
    
    return text.strip()


def get_chunks(text_content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    
    try:
        chunk_size_chars = chunk_size * 4
        overlap_chars = chunk_overlap * 4
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_chars,
            chunk_overlap=overlap_chars,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(text_content)
        
        return chunks
        
    except Exception as e:
        print(f"❌ Error generando chunks: {str(e)}")
        raise ValueError(f"Error en chunking: {str(e)}")


def get_embeddings(chunks: List[str], model_id: str = "amazon.titan-embed-text-v2:0", dimensions: int = 1024) -> List[List[float]]:

    if not chunks:
        raise ValueError("La lista de chunks no puede estar vacía")
    
    if dimensions not in [1024, 512, 256]:
        raise ValueError("Dimensiones soportadas por Titan V2: 1024, 512, 256")
    
    try:
 
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

        embeddings = []
        
        for i, chunk in enumerate(chunks):
            try:
                print(f"🔄 Procesando chunk {i+1}/{len(chunks)} - {len(chunk)} caracteres")
                
                payload = {
                    "inputText": chunk.strip(),
                    "dimensions": dimensions,
                    "normalize": True
                }
            
                response = bedrock_runtime.invoke_model(
                    modelId=model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(payload)
                )
                
                response_body = json.loads(response['body'].read())
                embedding = response_body.get('embedding', [])
                
                if not embedding:
                    print(f"❌ Bedrock no devolvió embedding para chunk {i+1}")
                    continue
                    
                embeddings.append(embedding)
                print(f"✅ Chunk {i+1} procesado - {len(embedding)} dimensiones")
                
            except Exception as chunk_error:
                print(f"❌ Error en chunk {i+1}: {str(chunk_error)}")
                continue
        
        print(f"🎉 Embeddings generados: {len(embeddings)} vectores de {dimensions} dimensiones")
        return embeddings
        
    except Exception as e:
        print(f"❌ Error en cliente Bedrock: {str(e)}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Error generando embeddings: {str(e)}")


def get_embedding_dimensions(model_id: str = "amazon.titan-embed-text-v2:0") -> int:
    """
    Devuelve las dimensiones por defecto del modelo de embedding
    
    Args:
        model_id: Modelo de Bedrock
    
    Returns:
        Número de dimensiones del vector por defecto
    """
    model_dimensions = {
        "amazon.titan-embed-text-v1:0": 1536,  # Fijo
        "amazon.titan-embed-text-v2:0": 1024,  # Default configurable: 1024, 512, 256
    }
    
    return model_dimensions.get(model_id, 1024)


def generate_document_hash(tenant_id: str, source_file: str, chunk_index: int, content_sample: str) -> str:
    """
    Genera un hash único para identificar un documento/chunk específico
    
    Args:
        tenant_id: ID del tenant
        source_file: Ruta del archivo fuente
        chunk_index: Índice del chunk
        content_sample: Muestra del contenido para mayor unicidad
    
    Returns:
        Hash MD5 como string hexadecimal
    """
    # Combinar elementos únicos
    unique_string = f"{tenant_id}|{source_file}|{chunk_index}|{content_sample}"
    
    # Generar hash MD5
    hash_object = hashlib.md5(unique_string.encode('utf-8'))
    document_hash = hash_object.hexdigest()
    
    return document_hash


def create_opensearch_client(region: str = 'us-east-1') -> OpenSearch:

    try:
        print(f"🔐 Inicializando cliente OpenSearch para región {region}")
        
        session = boto3.Session()
        credentials = session.get_credentials()
        
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            'aoss',  # Amazon OpenSearch Serverless
            session_token=credentials.token
        )
        
        opensearch_endpoint = os.environ.get('OPENSEARCH_ENDPOINT')
        if not opensearch_endpoint:
            raise ValueError("Variable OPENSEARCH_ENDPOINT no configurada")
        
        host = opensearch_endpoint
        
        # Crear cliente
        client = OpenSearch(
            hosts=[{'host': host.replace('https://', ''), 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=60
        )
        
        print(f"✅ Cliente OpenSearch creado exitosamente")
        return client
        
    except Exception as e:
        print(f"❌ Error creando cliente OpenSearch: {str(e)}")
        raise ValueError(f"Error en cliente OpenSearch: {str(e)}")


def create_index_if_not_exists(
    client: OpenSearch, 
    index_name: str, 
    dimensions: int = 1024
) -> bool:

    try:

        if client.indices.exists(index=index_name):
            print(f"📋 Índice '{index_name}' ya existe")
            return True
        
        print(f"🆕 Creando índice '{index_name}' con {dimensions} dimensiones")
        
        index_mapping = {
            "settings": {
                "index": {
                    "knn": True,  # Habilitar k-NN search
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "tenant_id": {
                        "type": "keyword"  # Para filtrado exacto
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dimensions,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    },
                    "document_type": {
                        "type": "keyword"
                    },
                    "file_format": {
                        "type": "keyword"
                    },
                    "source_file": {
                        "type": "keyword"
                    },
                    "chunk_index": {
                        "type": "integer"
                    },
                    "created_at": {
                        "type": "date",
                        "format": "strict_date_optional_time"
                    },
                    "content_type": {
                        "type": "keyword"  # "text" o "image"
                    },
                    "description": {
                        "type": "text",
                        "analyzer": "standard"  # Para imágenes principalmente
                    }
                }
            }
        }
        
        # Crear índice
        response = client.indices.create(
            index=index_name,
            body=index_mapping
        )
        
        print(f"✅ Índice '{index_name}' creado exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error creando índice '{index_name}': {str(e)}")
        return False


def index_document_bulk(
    client: OpenSearch,
    index_name: str,
    documents: List[Dict],
    tenant_id: str
) -> bool:

    try:
        if not documents:
            print("No hay documentos para indexar")
            return True
        
        print(f"Preparando bulk indexing de {len(documents)} documentos para tenant '{tenant_id}'")
        
        bulk_body = []
        timestamp = datetime.utcnow().isoformat()
        
        for i, doc in enumerate(documents):
            action = {
                "index": {
                    "_index": index_name
                }
            }
            bulk_body.append(action)
            
            content_hash = generate_document_hash(
                tenant_id, 
                doc.get('source_file', 'unknown'), 
                i,
                doc['content'][:100]  # Primeros 100 chars del contenido
            )
            
            # Documento completo
            document = {
                "tenant_id": tenant_id,
                "content": doc['content'],
                "embedding": doc['embedding'],
                "document_type": doc.get('document_type', 'unknown'),
                "file_format": doc.get('file_format', 'unknown'),
                "source_file": doc.get('source_file', 'unknown'),
                "chunk_index": i,
                "document_hash": content_hash,  # Hash único para identificación
                "created_at": timestamp
            }
            bulk_body.append(document)
        
        # Ejecutar bulk request
        print(f"🚀 Ejecutando bulk indexing...")
        response = client.bulk(body=bulk_body)
        
        # Verificar errores
        if response.get('errors'):
            failed_docs = []
            for item in response['items']:
                if 'error' in item.get('index', {}):
                    error_info = item['index']['error']
                    failed_docs.append(f"ID: {item['index']['_id']}, Error: {error_info}")
                    print(f"❌ Error indexando: {error_info}")
            
            if failed_docs:
                print(f"⚠️ {len(failed_docs)} documentos fallaron en indexado")
                return False
        
        successful_docs = len([item for item in response['items'] if 'error' not in item.get('index', {})])
        print(f"✅ Bulk indexing completado: {successful_docs}/{len(documents)} documentos indexados")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en bulk indexing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def get_multimodal_embeddings(base64_image: str = None, input_text: str = None, dimensions: int = 1024) -> List[List[float]]:

    if dimensions not in [1024, 384, 256]:
        raise ValueError("Dimensiones soportadas por Titan Multimodal: 1024, 384, 256")
    
    if not base64_image and not input_text:
        raise ValueError("Debe proporcionar al menos base64_image o input_text")
    
    try:
        
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        # Determinar tipo de embedding
        if base64_image and input_text:
            print(f"🖼️🔤 Generando embedding multimodal (imagen + texto) - {dimensions} dimensiones")
        elif base64_image:
            print(f"🖼️ Generando embedding de imagen - {dimensions} dimensiones")
        else:
            print(f"🔤 Generando embedding de texto (modelo multimodal) - {dimensions} dimensiones")
        
        # Payload para Titan Multimodal
        payload = {
            "embeddingConfig": {
                "outputEmbeddingLength": dimensions
            }
        }
        
        # Agregar imagen si está presente
        if base64_image:
            payload["inputImage"] = base64_image
            
        # Agregar texto si está presente
        if input_text:
            payload["inputText"] = input_text.strip()
            if len(input_text) > 50:
                print(f"📝 Texto: {input_text[:50]}...")
            else:
                print(f"📝 Texto: {input_text}")
        
        response = bedrock_runtime.invoke_model(
            modelId="amazon.titan-embed-image-v1",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        
        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding', [])
        
        if not embedding:
            print("❌ Titan Multimodal no devolvió embedding")
            raise ValueError("No se pudo generar embedding multimodal")
            
        print(f"✅ Embedding multimodal generado - {len(embedding)} dimensiones")
        return [embedding]  # Devolver como lista para mantener consistencia
        
    except Exception as e:
        print(f"❌ Error generando embedding multimodal: {str(e)}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Error en embedding multimodal: {str(e)}")



