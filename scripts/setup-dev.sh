#!/bin/bash

# ================================
# Setup Script para Desenvolvimento
# Agent Core Enhanced
# ================================

set -e  # Exit on any error

echo "🚀 Configurando ambiente de desenvolvimento do Agent Core Enhanced..."

# ================================
# Verificar Python
# ================================

echo "🐍 Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Instale Python 3.11+ primeiro."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "✅ Python $PYTHON_VERSION encontrado"

# ================================
# Criar ambiente virtual
# ================================

echo "🔧 Configurando ambiente virtual..."

if [ ! -d "venv" ]; then
    echo "📦 Criando ambiente virtual..."
    python3 -m venv ../venv
else
    echo "✅ Ambiente virtual já existe"
fi

# Ativar ambiente virtual
echo "🔥 Ativando ambiente virtual..."
source ../venv/bin/activate

# ================================
# Instalar dependências
# ================================

echo "📚 Instalando dependências..."

# Atualizar pip
pip install --upgrade pip setuptools wheel

# Instalar dependências principais
echo "📦 Instalando dependências do requirements.txt..."
pip install -r ../requirements.txt

echo "✅ Dependências instaladas com sucesso"

# ================================
# Configurar arquivo de ambiente
# ================================

echo "⚙️ Configurando arquivo de ambiente..."

if [ ! -f ".env" ]; then
    echo "📝 Criando arquivo .env a partir do template..."
    cp ../.env.example .env
    echo "✅ Arquivo .env criado. IMPORTANTE: Configure suas variáveis de ambiente!"
else
    echo "✅ Arquivo .env já existe"
fi

# ================================
# Criar diretórios necessários
# ================================

echo "📁 Criando diretórios necessários..."

directories=("logs" "eval_results" "chroma_db")
for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "✅ Diretório criado: $dir"
    else
        echo "✅ Diretório já existe: $dir"
    fi
done

# ================================
# Testar instalação
# ================================

echo "🧪 Testando instalação..."

echo "📋 Verificando módulos principais..."
python3 -c "
import sys
modules = [
    'langgraph',
    'langchain',
    'mcp',
    'fastmcp',
    'requests',
    'dotenv'
]

for module in modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError as e:
        print(f'❌ {module}: {e}')
        sys.exit(1)
"

# Testar funcionalidades aprimoradas
echo "🔍 Verificando funcionalidades aprimoradas..."
python3 -c "
import sys
import os
sys.path.insert(0, '.')

enhanced_modules = [
    'utils.feature_toggles',
    'utils.enhanced_logging',
    'utils.oauth2_auth',
    'utils.advanced_reasoning',
    'utils.prompt_evals'
]

for module in enhanced_modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError as e:
        print(f'⚠️ {module}: {e}')
"

# ================================
# Executar testes rápidos
# ================================

echo "🧪 Executando testes de integração..."
if python3 integration_test.py; then
    echo "✅ Testes de integração passaram!"
else
    echo "⚠️ Alguns testes falharam, mas o setup foi concluído."
fi

# ================================
# Finalização
# ================================

echo ""
echo "🎉 Setup concluído com sucesso!"
echo ""
echo "📋 Próximos passos:"
echo "1. Configure suas variáveis no arquivo .env"
echo "2. Para desenvolvimento local:"
echo "   source venv/bin/activate"
echo "   python mcp_server_enhanced.py"
echo ""
echo "3. Para testes:"
echo "   python integration_test.py"
echo ""
echo "4. Para MCP dev mode:"
echo "   mcp dev mcp_server_enhanced.py"
echo ""
echo "📚 Consulte o README.md para mais informações"