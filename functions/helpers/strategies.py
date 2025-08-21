from helpers.rag_helpers import extract_pdf_text, get_chunks, get_embeddings

def pdf_strategy(text):

    try:

        text_content = extract_pdf_text(text)

        if not text_content.strip():
            return {
                "success": False,
            "message": "No se pudo extraer texto"
        }

        chunks = get_chunks(text_content, 2000, 200)

        embeddings = get_embeddings(chunks, model_id="amazon.titan-embed-text-v2:0", dimensions=1024)

        return (chunks, embeddings)
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Error en pdf_strategy: {str(e)}"
        }

