from aws_cdk import (
    aws_opensearchserverless as opensearchserverless,
)
import json

def create_opensearch(app, prefix):

    network_policy = opensearchserverless.CfnSecurityPolicy(
        app, f"{prefix}-network-policy",
        name=f"{prefix}-network-policy",
        type="network",
        policy=json.dumps([
            {
                "Rules": [
                    {
                        "ResourceType": "collection",
                        "Resource": [f"collection/{prefix}-vector-collection"]
                    }
                ],
                "AllowFromPublic": True
            }
        ])
    )

    encryption_policy = opensearchserverless.CfnSecurityPolicy(
        app, f"{prefix}-encryption-policy", 
        name=f"{prefix}-encryption-policy",
        type="encryption",
        policy=json.dumps({
            "Rules": [
                {
                    "Resource": [f"collection/{prefix}-vector-collection"],
                    "ResourceType": "collection"
                }
            ],
            "AWSOwnedKey": True  # Usar AWS managed key
        })
    )

    vector_collection = opensearchserverless.CfnCollection(
        app, f"{prefix}-vector-collection",
        name=f"{prefix}-vector-collection",
        type="VECTORSEARCH",  # Optimizada para búsqueda vectorial
        description="Multi-tenant RAG vector search collection"
    )

    vector_collection.add_dependency(network_policy)
    vector_collection.add_dependency(encryption_policy)


    return vector_collection