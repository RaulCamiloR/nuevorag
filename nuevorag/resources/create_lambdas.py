from aws_cdk import (
    Duration,  
    aws_lambda as lambda_,
    aws_iam as iam,
)

from aws_cdk.aws_lambda_python_alpha import PythonFunction


def create_test_lambda(app, prefix, layer):

    test_lambda = PythonFunction(app, f"{prefix}-TestLambda",
        runtime=lambda_.Runtime.PYTHON_3_12,
        entry="functions",  
        handler="lambda_handler",    
        index="test.py",           
        layers=[layer],    
        timeout=Duration.minutes(5), 
        memory_size=1024,         
    )

    test_lambda.add_to_role_policy(
        iam.PolicyStatement(
        effect=iam.Effect.ALLOW,
        actions=[
            "bedrock:InvokeModel",
            "bedrock:ListFoundationModels",
            "bedrock:GetFoundationModel"
        ],
        resources=["*"]  # En producción, especifica ARNs específicos
        )
    )

    return test_lambda


def create_process_lambda(app, prefix, layer):
    """
    Crea la Lambda para procesar archivos subidos a S3
    """
    
    process_lambda = PythonFunction(app, f"{prefix}-ProcessLambda",
        runtime=lambda_.Runtime.PYTHON_3_12,
        entry="functions",  
        handler="lambda_handler",    
        index="process.py",           
        layers=[layer],    
        timeout=Duration.minutes(15),  # Más tiempo para procesamiento
        memory_size=2048,              # Más memoria para procesar archivos grandes
    )

    # Permisos para leer de S3
    process_lambda.add_to_role_policy(
        iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "s3:GetObject",
                "s3:GetObjectVersion"
            ],
            resources=["*"]  # En producción, especificar bucket específico
        )
    )
    
    # Permisos para Bedrock (para futuras funcionalidades)
    process_lambda.add_to_role_policy(
        iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "bedrock:InvokeModel",
                "bedrock:ListFoundationModels",
                "bedrock:GetFoundationModel"
            ],
            resources=["*"]
        )
    )

    return process_lambda

def create_upload_lambda(app, prefix, layer, bucket):
    """
    Crea la Lambda para generar presigned URLs para upload de archivos
    """
    
    upload_lambda = PythonFunction(app, f"{prefix}-UploadLambda",
        runtime=lambda_.Runtime.PYTHON_3_12,
        entry="functions",  
        handler="lambda_handler",    
        index="upload.py",           
        layers=[layer],    
        timeout=Duration.minutes(1),   # Operación rápida
        memory_size=512,               # Poca memoria necesaria
        environment={
            "BUCKET_NAME": bucket.bucket_name  # Pasar nombre del bucket
        }
    )

    # Permisos para generar presigned URLs (solo lectura de metadatos)
    upload_lambda.add_to_role_policy(
        iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "s3:PutObject",
                "s3:PutObjectAcl"
            ],
            resources=[f"{bucket.bucket_arn}/uploads/*"]  # Solo en carpeta uploads
        )
    )
    
    # Permisos para acceder a metadatos del bucket
    upload_lambda.add_to_role_policy(
        iam.PolicyStatement(
            effect=iam.Effect.ALLOW,
            actions=[
                "s3:GetBucketLocation"
            ],
            resources=[bucket.bucket_arn]
        )
    )

    return upload_lambda