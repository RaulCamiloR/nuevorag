import json
import urllib.parse
import boto3
import os
from helpers.rag_helpers import extract_pdf_text, get_chunks, get_embeddings

def lambda_handler(event, context):
    
    s3_client = boto3.client('s3')
    
    for record in event.get('Records', []):
        try:
            event_name = record.get('eventName', '')
            bucket_name = record['s3']['bucket']['name']
            object_key = urllib.parse.unquote_plus(
                record['s3']['object']['key'], 
                encoding='utf-8'
            )
            object_size = record['s3']['object']['size']
            
            path_parts = object_key.split('/')
            if len(path_parts) < 3:
                print(f"❌ Path inválido: {object_key}")
                continue
                
            tenant_id = path_parts[1] if path_parts[0] == 'uploads' else 'unknown'
            document_type = path_parts[2] if len(path_parts) >= 3 else 'general'
            filename = path_parts[-1]
            
            extension = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
            
            print(f"Tenant ID: {tenant_id}")
            print(f"Tipo documento: {document_type}")
            print(f"Nombre archivo: {filename}")
            print(f"Extensión: {extension}")
            
            if extension == '.pdf':
                result = process_pdf_file(
                    s3_client, bucket_name, object_key, 
                    tenant_id, document_type, filename
                )
                print(f"Resultado procesamiento PDF: {result}")
            else:
                print(f"⚠️ Extensión {extension} no soportada aún")
                continue
                
        except Exception as e:
            print(f"❌ Error procesando archivo {object_key}: {str(e)}")
            continue
        
        print("=== FIN PROCESAMIENTO ===")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Procesados {len(event.get("Records", []))} archivos exitosamente'
        })
    }


def process_pdf_file(s3_client, bucket_name, object_key, tenant_id, document_type, filename):
    
    try:
        
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        file_content = response['Body'].read()
        
        text_content = extract_pdf_text(file_content)
        
        if not text_content.strip():
            return {
                "success": False,
                "message": "No se pudo extraer texto del PDF"
            }
        
        # estrategia depende del tipo de documento con IF
        # importante futuro!!!
        
        chunks = get_chunks(text_content, 2000, 200)
        
        embeddings = get_embeddings(chunks, model_id="amazon.titan-embed-text-v1:0", dimensions=1536)
        
        # TODO: Indexar en OpenSearch (próximo paso)
        print(f"📋 TODO: Indexar embeddings en OpenSearch")
        
        print(f"=== PROCESAMIENTO PDF EXITOSO ===")
        return {
            "success": True,
            "message": "PDF procesado exitosamente",
            "stats": {
                "characters": len(text_content),
                "chunks": len(chunks),
                "embeddings": len(embeddings),
                "dimensions": len(embeddings[0]) if embeddings else 0
            }
        }
        
    except Exception as e:
        print(f"❌ Error procesando PDF: {str(e)}")
        return {
            "success": False,
            "message": f"Error procesando PDF: {str(e)}"
        }