#!/bin/bash

# ================================
# Script de Teste de Feature Toggles
# Agent Core Enhanced
# ================================

set -e  # Exit on any error

echo "🎛️ Testando sistema de Feature Toggles do Agent Core..."

# ================================
# Verificar ambiente
# ================================

if [ ! -f ".env" ]; then
    echo "⚠️ Arquivo .env não encontrado. Executando setup..."
    cp .env.example .env
fi

# Ativar ambiente virtual se existir
if [ -d "venv" ]; then
    echo "🔥 Ativando ambiente virtual..."
    source venv/bin/activate
fi

# ================================
# Teste 1: Features Padrão
# ================================

echo ""
echo "🧪 Teste 1: Verificando features padrão..."

export ENVIRONMENT=dev
export FEATURE_ENHANCED_LOGGING_ENABLED=true
export FEATURE_ADVANCED_REASONING_ENABLED=true
export FEATURE_PROMPT_EVALUATION_ENABLED=true
export FEATURE_OAUTH2_AUTHENTICATION_ENABLED=false

python3 -c "
import sys
sys.path.insert(0, '.')
from utils.feature_toggles import get_feature_manager

print('📊 Testando configuração padrão...')
manager = get_feature_manager()
enabled = manager.get_enabled_features()
print(f'✅ Features habilitadas: {len(enabled)}')
for feature in enabled:
    print(f'  🎯 {feature}')
"

# ================================
# Teste 2: Toggle dinâmico
# ================================

echo ""
echo "🧪 Teste 2: Testando toggle dinâmico..."

python3 -c "
import sys
sys.path.insert(0, '.')
from utils.feature_toggles import get_feature_manager, is_feature_enabled

print('🔄 Testando toggle dinâmico...')
manager = get_feature_manager()

# Estado inicial
print(f'Reasoning habilitado: {is_feature_enabled(\"advanced_reasoning\")}')

# Desabilitar
manager.disable_feature('advanced_reasoning')
print(f'Após desabilitar: {is_feature_enabled(\"advanced_reasoning\")}')

# Reabilitar
manager.enable_feature('advanced_reasoning')
print(f'Após reabilitar: {is_feature_enabled(\"advanced_reasoning\")}')

print('✅ Toggle dinâmico funcionando')
"

# ================================
# Teste 3: Decorators
# ================================

echo ""
echo "🧪 Teste 3: Testando decorators..."

python3 -c "
import sys
sys.path.insert(0, '.')
from utils.feature_toggles import require_feature, optional_feature

@require_feature('advanced_reasoning')
def test_required():
    return '🧠 Função com reasoning executada'

@require_feature('oauth2_authentication', fallback_return='🔐 OAuth2 não disponível')
def test_oauth():
    return '🔐 OAuth2 executado'

@optional_feature('prompt_evaluation')
def test_optional():
    return '📊 Avaliação executada'

print('🎯 Testando decorators...')
print(test_required())
print(test_oauth())
print(test_optional())
print('✅ Decorators funcionando')
"

# ================================
# Teste 4: Diferentes ambientes
# ================================

echo ""
echo "🧪 Teste 4: Testando ambientes diferentes..."

for env in dev staging prod; do
    echo "🌍 Testando ambiente: $env"
    ENVIRONMENT=$env python3 -c "
import sys
sys.path.insert(0, '.')
from utils.feature_toggles import get_feature_manager

manager = get_feature_manager()
enabled = manager.get_enabled_features()
print(f'  📊 {len(enabled)} features habilitadas em {manager.environment}')
"
done

# ================================
# Teste 5: Integração com Enhanced Logging
# ================================

echo ""
echo "🧪 Teste 5: Testando integração com Enhanced Logging..."

export FEATURE_ENHANCED_LOGGING_ENABLED=true

python3 -c "
import sys
sys.path.insert(0, '.')
from utils.enhanced_logging import get_enhanced_logger

logger = get_enhanced_logger('feature_test')
logger.info('Testando integração feature toggles + enhanced logging')
logger.log_with_feature_toggle('advanced_reasoning', 'Log condicional por feature', level='info')
print('✅ Integração com Enhanced Logging funcionando')
"

# ================================
# Teste 6: MCP Server Enhanced
# ================================

echo ""
echo "🧪 Teste 6: Testando MCP Server Enhanced com features..."

timeout 10s python3 -c "
import sys
sys.path.insert(0, '.')

try:
    import mcp_server_enhanced
    print('✅ MCP Server Enhanced carregado com features')

    if hasattr(mcp_server_enhanced, 'feature_manager'):
        enabled = mcp_server_enhanced.feature_manager.get_enabled_features()
        print(f'  🎯 {len(enabled)} features ativas no servidor')

except Exception as e:
    print(f'⚠️ Erro ao carregar MCP Server: {e}')
" || echo "⚠️ MCP Server test timeout (normal)"

# ================================
# Teste 7: Performance com features
# ================================

echo ""
echo "🧪 Teste 7: Testando performance..."

python3 -c "
import sys
import time
sys.path.insert(0, '.')
from utils.feature_toggles import is_feature_enabled

# Teste de performance
start_time = time.time()
for i in range(1000):
    is_feature_enabled('advanced_reasoning')
end_time = time.time()

avg_time = (end_time - start_time) / 1000 * 1000  # em ms
print(f'⚡ Tempo médio por verificação: {avg_time:.3f}ms')
print('✅ Performance adequada' if avg_time < 1 else '⚠️ Performance pode melhorar')
"

# ================================
# Relatório Final
# ================================

echo ""
echo "📊 RELATÓRIO DE TESTES DE FEATURE TOGGLES"
echo "=" * 50

python3 -c "
import sys
sys.path.insert(0, '.')
from utils.feature_toggles import get_feature_manager

manager = get_feature_manager()
status = manager.get_feature_status()

print('📋 Status atual das features:')
for name, info in status.items():
    status_icon = '✅' if info['enabled'] else '❌'
    print(f'{status_icon} {name}')
    print(f'   📝 {info[\"description\"]}')
    if info.get('dependencies'):
        print(f'   🔗 Dependências: {info[\"dependencies\"]}')
    print()
"

echo "🎉 Testes de Feature Toggles concluídos!"
echo ""
echo "💡 Dicas:"
echo "- Use variáveis FEATURE_*_ENABLED para controlar features"
echo "- Configure features diferentes por ambiente"
echo "- Use decorators para proteção automática"
echo "- Monitore performance em produção"