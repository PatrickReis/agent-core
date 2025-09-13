# ================================
# Multi-stage Dockerfile for Agent Core
# Suporta tanto desenvolvimento quanto produção/Lambda
# ================================

# ================================
# Stage 1: Base Image
# ================================
FROM python:3.11-slim as base

# Configurar variáveis de ambiente para Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root para segurança
RUN useradd --create-home --shell /bin/bash agent
WORKDIR /home/agent/app
RUN chown -R agent:agent /home/agent

# ================================
# Stage 2: Dependencies
# ================================
FROM base as dependencies

# Copiar arquivos de requirements primeiro (para cache de layers)
COPY requirements.txt ./
COPY requirements-lambda.txt ./

# Instalar dependências Python
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# ================================
# Stage 3: Development
# ================================
FROM dependencies as development

USER agent

# Copiar código fonte
COPY --chown=agent:agent . .

# Instalar dependências de desenvolvimento
RUN pip install --user pytest black isort flake8 mypy

# Expor porta para desenvolvimento
EXPOSE 8000

# Comando padrão para desenvolvimento
CMD ["python", "mcp_server_enhanced.py"]

# ================================
# Stage 4: Production
# ================================
FROM dependencies as production

USER agent

# Copiar apenas arquivos necessários para produção
COPY --chown=agent:agent utils/ ./utils/
COPY --chown=agent:agent tools/ ./tools/
COPY --chown=agent:agent graphs/ ./graphs/
COPY --chown=agent:agent providers/ ./providers/
COPY --chown=agent:agent transform/ ./transform/
COPY --chown=agent:agent logger/ ./logger/
COPY --chown=agent:agent mcp_server_enhanced.py ./
COPY --chown=agent:agent lambda_handler.py ./
COPY --chown=agent:agent feature_toggles.json ./
COPY --chown=agent:agent .env.example ./

# Criar diretórios necessários
RUN mkdir -p logs eval_results chroma_db

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando padrão para produção
CMD ["python", "mcp_server_enhanced.py"]

# ================================
# Stage 5: Lambda
# ================================
FROM public.ecr.aws/lambda/python:3.11 as lambda

# Copiar dependências Lambda-específicas
COPY requirements-lambda.txt .
RUN pip install -r requirements-lambda.txt

# Copiar código otimizado para Lambda
COPY utils/ ${LAMBDA_TASK_ROOT}/utils/
COPY tools/ ${LAMBDA_TASK_ROOT}/tools/
COPY graphs/ ${LAMBDA_TASK_ROOT}/graphs/
COPY providers/ ${LAMBDA_TASK_ROOT}/providers/
COPY transform/ ${LAMBDA_TASK_ROOT}/transform/
COPY logger/ ${LAMBDA_TASK_ROOT}/logger/
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/
COPY feature_toggles.json ${LAMBDA_TASK_ROOT}/

# Configurar handler Lambda
CMD ["lambda_handler.lambda_handler"]

# ================================
# Build Arguments e Labels
# ================================
ARG BUILD_DATE
ARG VERSION=2.0.0
ARG VCS_REF

LABEL maintainer="Agent Core Team" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="agent-core-enhanced" \
      org.label-schema.description="Enhanced Agent Core with feature toggles, OAuth2, and advanced reasoning" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.schema-version="1.0"

# ================================
# USAGE INSTRUCTIONS
# ================================
#
# Build for development:
# docker build --target development -t agent-core:dev .
#
# Build for production:
# docker build --target production -t agent-core:prod .
#
# Build for Lambda:
# docker build --target lambda -t agent-core:lambda .
#
# Run development:
# docker run -p 8000:8000 -e ENVIRONMENT=dev agent-core:dev
#
# Run production:
# docker run -p 8000:8000 -e ENVIRONMENT=prod agent-core:prod
#
# Run with custom environment file:
# docker run -p 8000:8000 --env-file .env agent-core:prod
#
# For Lambda, deploy using AWS CLI or CDK:
# aws lambda create-function --function-name agent-core \
#   --code ImageUri=your-account.dkr.ecr.region.amazonaws.com/agent-core:lambda \
#   --role arn:aws:iam::your-account:role/lambda-execution-role \
#   --timeout 300 --memory-size 1024
#