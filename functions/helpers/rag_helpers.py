"""
RAG Helpers - Funciones para procesamiento de documentos y generación de embeddings
"""
import boto3
import json
import io
import PyPDF2
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
# BedrockEmbeddings removido - ahora usamos boto3 directo


def extract_pdf_text(file_content: bytes) -> str:
    
    try:
        pdf_file = io.BytesIO(file_content)
        
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        print(f"📄 PDF tiene {len(pdf_reader.pages)} páginas")
        
        text_content = ""

        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                page_text = page.extract_text()
                text_content += f"\n{page_text}\n"
                print(f"📄 Página {page_num}: {len(page_text)} caracteres")
            except Exception as e:
                print(f"⚠️ Error en página {page_num}: {str(e)}")
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
 
        print(f"🔢 Inicializando cliente Bedrock Runtime...")
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        print(f"✅ Cliente inicializado - Modelo: {model_id}, Dimensiones: {dimensions}")
        
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            try:
                print(f"🔄 Procesando chunk {i+1}/{len(chunks)} - {len(chunk)} caracteres")
                
                # Payload para Titan V2
                payload = {
                    "inputText": chunk.strip(),
                    "dimensions": dimensions,
                    "normalize": True
                }
                
                # Llamada a Bedrock
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

