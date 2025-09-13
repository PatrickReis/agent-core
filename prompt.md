Você é um especialista em desenvolvimento de sistemas de IA e vai implementar melhorias em um projeto Agent Core existente. Este projeto serve como archetype/template base para criação de novos agentes LangGraph e servidores MCP.

## 📋 OBJETIVO PRINCIPAL
Implementar 4 funcionalidades principais mantendo o projeto como template reutilizável:

1. **Autenticação OAuth2** para MCP Server
2. **Sistema de Logging Estruturado** com métricas
3. **Sistema de Feature Toggles** (controle de funcionalidades)  
4. **Sistema de Reasoning Avançado** (multi-step com reflexão)
5. **Utilitários**: Migração de tools e sistema de evals como utilities

## 🎯 ESTRUTURA DE ARQUIVOS A CRIAR/MODIFICAR

### 1. Sistema de Feature Toggles
**CRIAR: `utils/feature_toggles.py`**
- Sistema completo de feature toggles
- Decorators @require_feature, @optional_feature
- Configuração via env vars e JSON
- Rollout percentage, user whitelist, dependencies
- Gerenciamento por ambiente (dev/staging/prod)

### 2. Sistema de Autenticação OAuth2  
**CRIAR: `utils/oauth2_auth.py`**
- Middleware OAuth2 para MCP
- Validação JWT com JWKS
- Cliente OAuth2 para service-to-service
- Proteção de tools por scope
- Configuração via secrets manager

### 3. Logging Estruturado
**CRIAR: `utils/enhanced_logging.py`** 
- Substituir sistema de logging atual
- Logs estruturados em JSON
- Métricas automáticas (latência, success rate, etc)
- Context variables para tracing distribuído
- Múltiplos handlers (console, arquivo, métricas)
- TimingContext para medição automática

### 4. Sistema de Reasoning Avançado
**CRIAR: `utils/advanced_reasoning.py`**
- Reasoning multi-step com análise, planejamento, execução
- Reflexão automática e adaptação dinâmica  
- Traces completos de raciocínio
- Orquestração inteligente baseada na complexidade
- Integração com feature toggles

### 5. Sistema de Evals (Utility)
**CRIAR: `utils/prompt_evals.py`**
- Framework de avaliação de prompts
- Múltiplas métricas (relevância, acurácia, latência, segurança)
- Suites de teste configuráveis
- Relatórios com agregações estatísticas
- Integração com feature toggles
- Builder pattern para criação de suites

### 6. Migração de Tools (Utility)
**MOVER: `utils/openapi_to_tools.py` (já existe - manter e melhorar)**
- Converter OpenAPI para LangGraph Tools
- Geração automática de código Python
- Suporte a múltiplos métodos HTTP
- Validação de parâmetros

### 7. MCP Server Aprimorado
**CRIAR: `mcp_server_enhanced.py`**
- Integração de todas as funcionalidades
- Proteção OAuth2 condicional
- Tools de gerenciamento de features
- Resources para status e métricas
- Prompts context-aware

### 8. Handler Lambda
**CRIAR: `lambda_handler.py`** 
- Handler otimizado para AWS Lambda
- Cold start mitigation
- Validação de requests
- Integração com todas as features
- Health checks

### 9. Arquivos de Configuração
**CRIAR múltiplos arquivos de config:**
- `.env.example` - atualizado com todas as configs
- `feature_toggles.json` - configuração padrão das features
- `requirements-lambda.txt` - deps específicas do Lambda
- `Dockerfile` - container otimizado para Lambda

## 🔧 DIRETRIZES DE IMPLEMENTAÇÃO

### **Feature Toggles - Regras Importantes:**
- Sistema deve ser **não-intrusivo** - código funciona mesmo sem feature toggles
- **Graceful degradation** - features desabilitadas não quebram o sistema
- **Logs claros** quando features estão desabilitadas
- **Configuração via env vars** tem precedência sobre arquivo JSON
- **Decorators automáticos** para proteger funções

### **Archetype/Template - Características:**
- **Código comentado** explicando cada funcionalidade
- **Configuração flexível** para diferentes tipos de projetos
- **Documentação inline** para facilitar customização
- **Componentes modulares** que podem ser removidos se não necessários
- **Exemplos de uso** em cada arquivo

### **Integração entre Componentes:**
- Feature toggles controlam **OAuth2, Evals, Reasoning, Monitoring**
- Enhanced logging usado por **todos os componentes**
- MCP server integra **todas as funcionalidades de forma condicional**
- Lambda handler **otimizado** para serverless

### **Configurações por Ambiente:**
- **DEV**: Todas features habilitadas para desenvolvimento
- **STAGING**: Features de produção + evals para validação  
- **PROD**: Apenas features essenciais (OAuth, Reasoning, Logging)

## 📝 IMPLEMENTAÇÃO ESPECÍFICA
```python

1. Feature Toggles (`utils/feature_toggles.py`)
# Features padrão a implementar:
- prompt_evaluation (dev/staging only)
- advanced_reasoning (all environments)  
- oauth2_authentication (staging/prod only)
- enhanced_logging (all environments)
- performance_monitoring (all environments)
- experimental_features (dev only, disabled)

# Funcionalidades requeridas:
- FeatureToggleManager class
- @require_feature e @optional_feature decorators
- Suporte a rollout percentage
- User whitelist
- Dependencies entre features
- Configuração via env vars FEATURE_<NAME>_ENABLED=true/false
2. Enhanced Logging (utils/enhanced_logging.py)
# Funcionalidades requeridas:
- EnhancedLogger class com métodos especializados
- TimingContext para medição automática
- Métricas automáticas (MetricsCollector)
- Context variables (request_id, user_id, session_id)
- Múltiplos formatters (JSON estruturado + console legível)
- Método operation_start/success/error para operações complexas
3. OAuth2 Auth (utils/oauth2_auth.py)
# Funcionalidades requeridas:
- OAuth2Config class
- JWKSCache para cache de chaves
- TokenValidator para JWT
- MCPAuthMiddleware para proteção de tools
- OAuth2Client para service-to-service
- Decorators @require_scope
- Configuração via secrets manager na AWS
4. Advanced Reasoning (utils/advanced_reasoning.py)
# Funcionalidades requeridas:
- AdvancedAgentState com reasoning_traces
- ReasoningAnalyzer para análise de complexidade
- ReasoningPlanner para planejamento multi-step
- ReasoningExecutor para execução de steps
- Múltiplos tipos de steps: ANALYZE, PLAN, EXECUTE, REFLECT, DECIDE
- Orquestração inteligente baseada na pergunta
- Integração com feature toggles
5. Prompt Evals (utils/prompt_evals.py)
# Funcionalidades requeridas:
- EvalFramework class principal
- Multiple evaluators: RelevanceEvaluator, AccuracyEvaluator, ToolUsageEvaluator, LatencyEvaluator, SafetyEvaluator
- EvalSuiteBuilder com pattern builder
- Integração completa com feature toggles
- Salvamento de resultados em JSON
- Funções de conveniência: create_standard_eval_suite, run_quick_eval
6. MCP Server Enhanced (mcp_server_enhanced.py)
# Integrações requeridas:
- Importar e usar todos os utils criados
- Proteção OAuth2 condicional baseada em feature toggles
- Registration de tools com/sem proteção baseado nas features
- MCP Resources: metrics://system, features://status, eval://results, auth://status  
- MCP Tools: run_agent_evaluation, get_reasoning_trace, manage_features
- Prompts context-aware baseados nas features habilitadas
- Health checks baseados nas features ativas
7. Configurações (.env.example)
# Adicionar todas as configurações:
- OAuth2 (OAUTH2_CLIENT_ID, OAUTH2_CLIENT_SECRET, etc)
- Feature toggles (FEATURE_*_ENABLED)
- Enhanced logging configs
- AWS/Lambda configs
- Comentários explicativos para cada seção

⚙️ MODIFICAÇÕES EM ARQUIVOS EXISTENTES
1. Atualizar mcp_server.py

Manter versão original como referência
Adicionar comentário apontando para mcp_server_enhanced.py

2. Atualizar graphs/graph.py

Importar enhanced_logging ao invés do logger atual
Manter compatibilidade com reasoning básico
Adicionar hooks para reasoning avançado quando habilitado

3. Atualizar tools/tools.py

Usar enhanced_logging
Adicionar metadata para cada tool (para OAuth2 scopes)
Manter compatibilidade total

4. Atualizar providers/llm_providers.py

Integrar com enhanced_logging
Adicionar métricas de LLM calls automáticas

5. Criar integration_test.py

Teste completo de todas as funcionalidades
Verificação de feature toggles
Validação de OAuth2 (se habilitado)
Teste de evals (se habilitado)
Teste de reasoning (básico vs avançado)

🚀 DEPLOYMENT E INFRA
1. Dockerfile

Container otimizado para Lambda
Multi-stage build
Configuração para cold start mitigation

2. requirements-lambda.txt

Dependências específicas para AWS Lambda
boto3, mangum, aws-lambda-powertools

3. Scripts básicos

scripts/setup-dev.sh - configuração inicial
scripts/test-features.sh - teste das feature toggles

📚 DOCUMENTAÇÃO
1. README.md atualizado

Seção sobre feature toggles
Guia de configuração por ambiente
Exemplos de uso como archetype

2. README_FEATURES.md

Documentação completa das feature toggles
Como usar como template
Guia de customização

🧪 TESTES E VALIDAÇÃO
Cada arquivo criado deve ter:

Exemplo de uso no if __name__ == "__main__":
Docstrings completas explicando funcionalidade
Tratamento de erros robusto
Logs informativos durante execução

Teste final esperado:

python mcp_server_enhanced.py - deve iniciar sem erros
python integration_test.py - deve passar em todos os testes
Feature toggles funcionando via env vars
MCP tools/resources funcionais

🎯 RESULTADO ESPERADO
Ao final, o projeto deve servir como template completo para:

Criação de novos agentes LangGraph
Implementação de servidores MCP
Deploy em AWS Lambda
Sistema observável e seguro
Funcionalidades controláveis via feature toggles

O desenvolvedor deve poder usar este projeto como base e facilmente:

Habilitar/desabilitar funcionalidades via configuração
Customizar comportamento para seu caso de uso
Deploy em produção com segurança
Monitorar e avaliar performance

IMPORTANTE: Mantenha sempre compatibilidade backwards - código existente deve continuar funcionando mesmo com as novas funcionalidades adicionadas.
Execute as modificações de forma incremental, testando cada componente antes de prosseguir para o próximo.