# Agent Core Enhanced 🚀

**Template completo para desenvolvimento de agentes LangGraph com funcionalidades empresariais avançadas.**

Este projeto implementa um agente inteligente usando LangGraph com acesso a uma base de conhecimento vetorial e integração MCP (Model Context Protocol), servindo como archetype/template base para criação de novos agentes LangGraph e servidores MCP com funcionalidades empresariais completas.

## ✨ Funcionalidades Principais

### 🎛️ **Sistema de Feature Toggles**
- Controle granular de funcionalidades via configuração
- Decorators `@require_feature` e `@optional_feature`
- Rollout percentage e user whitelist
- Configuração por ambiente (dev/staging/prod)
- Dependencies entre features

### 📝 **Enhanced Logging Estruturado**
- Logs estruturados em JSON com contexto
- Métricas automáticas (latência, success rate)
- Context variables para tracing distribuído
- TimingContext para medição automática
- Múltiplos handlers (console, arquivo, métricas)

### 🔐 **Autenticação OAuth2**
- Middleware OAuth2 para MCP Server
- Validação JWT com JWKS cache
- Cliente OAuth2 para service-to-service
- Proteção de tools por scopes
- Integração com AWS Secrets Manager

### 🧠 **Reasoning Avançado**
- Reasoning multi-step com reflexão
- Análise automática de complexidade
- Traces completos de raciocínio
- Orquestração inteligente baseada na pergunta
- Adaptação dinâmica do processo

### 📊 **Sistema de Avaliação (Evals)**
- Framework completo de avaliação de prompts
- Múltiplas métricas (relevância, acurácia, latência, segurança)
- Builder pattern para criação de suites
- Relatórios com agregações estatísticas
- Integração com feature toggles

### ⚡ **Deployment e Infra**
- **MCP Server Enhanced** com todas as funcionalidades
- **Lambda Handler** otimizado para AWS
- **Docker** multi-stage para dev/prod/lambda
- Cold start mitigation
- Health checks e monitoring

## 🏗️ Estrutura do Projeto

```
agent-core/
├── agent.py                        # Agente principal com interface de linha de comando
├── mcp_server.py                    # Servidor MCP básico
├── mcp_server_enhanced.py           # Servidor MCP com todas as funcionalidades avançadas
├── lambda_handler.py                # Handler otimizado para AWS Lambda
├── auto_tools.py                    # Script de automação para geração de tools da API
├── graphs/
│   ├── __init__.py
│   └── graph.py                     # Lógica isolada do LangGraph (grafo, nós, estados)
├── providers/
│   ├── __init__.py
│   └── llm_providers.py             # Sistema modular de provedores LLM
├── tools/
│   ├── __init__.py
│   ├── tools.py                     # Ferramentas de busca vetorial e utilitários
│   └── weather_tools.py             # Ferramentas meteorológicas
├── utils/
│   ├── __init__.py
│   ├── feature_toggles.py           # Sistema de feature toggles
│   ├── enhanced_logging.py          # Logging estruturado com métricas
│   ├── oauth2_auth.py               # Autenticação OAuth2 completa
│   ├── advanced_reasoning.py        # Reasoning multi-step
│   ├── prompt_evals.py              # Framework de avaliação
│   ├── eval_runner.py               # Execução individual de evals
│   ├── model_comparison.py          # Comparação de modelos
│   ├── eval_reporter.py             # Geração de relatórios
│   └── openapi_to_tools.py          # Utilitário para conversão OpenAPI → LangGraph Tools
├── transform/
│   ├── __init__.py
│   └── lang_mcp_transform.py        # Bridge entre LangChain e MCP
├── logger/
│   ├── __init__.py
│   └── logger.py                    # Sistema de logging colorido
├── tests/
│   ├── __init__.py
│   ├── integration_test.py          # Testes completos de integração
│   └── test_mcp_enhanced.py         # Testes do MCP Server Enhanced
├── examples/
│   └── compare_models.py            # Exemplo de comparação de modelos
├── .env.example                     # Exemplo de configurações (100+ variáveis)
├── feature_toggles.json             # Configuração de features
├── requirements.txt                 # Dependências do projeto
├── requirements-lambda.txt          # Deps específicas Lambda
└── Dockerfile                       # Multi-stage para todos ambientes
```

## 🚀 Quick Start

### 1. Setup Automático
```bash
# Clone e configure automaticamente
git clone <repo>
cd agent-core
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente
```bash
cp .env.example .env
# Edite .env com suas configurações
```

### 3. Configuração de Provedores LLM

**Para Ollama (padrão):**
```bash
# O arquivo .env já está configurado para Ollama
ollama serve
```

**Para OpenAI:**
```bash
# Editar .env
MAIN_PROVIDER=openai
OPENAI_API_KEY=sua_chave_aqui

# Instalar dependência
pip install langchain-openai
```

**Para Google Gemini:**
```bash
# Editar .env
MAIN_PROVIDER=gemini
GEMINI_API_KEY=sua_chave_aqui

# Instalar dependência
pip install langchain-google-genai
```

### 4. Testar e Executar

```bash
# Testar configuração
python tests/integration_test.py

# Executar o agente principal
python agent.py

# === SERVIDORES MCP ===

# Servidor MCP Básico (desenvolvimento/testes)
fastmcp run mcp_server.py

# Servidor MCP Enhanced (produção/funcionalidades completas)
python mcp_server_enhanced.py

# Servidor MCP com HTTP
fastmcp run mcp_server.py --transport streamable-http --host 127.0.0.1 --port 8088

# Enhanced com HTTP
fastmcp run mcp_server_enhanced.py --transport streamable-http --host 127.0.0.1 --port 8089
```

## ⚙️ Configuração de Features

Configure features via variáveis de ambiente ou arquivo `feature_toggles.json`:

```bash
# Habilitar/desabilitar features
FEATURE_ADVANCED_REASONING_ENABLED=true
FEATURE_PROMPT_EVALUATION_ENABLED=true
FEATURE_OAUTH2_AUTHENTICATION_ENABLED=false
FEATURE_ENHANCED_LOGGING_ENABLED=true

# Configuração por ambiente
ENVIRONMENT=dev  # dev, staging, prod, lambda
```

### Features Disponíveis:

| Feature | Descrição | Ambientes Padrão |
|---------|-----------|------------------|
| `advanced_reasoning` | Reasoning multi-step com reflexão | todos |
| `prompt_evaluation` | Sistema de avaliação de prompts | dev, staging |
| `oauth2_authentication` | Autenticação OAuth2 completa | staging, prod |
| `enhanced_logging` | Logging estruturado com métricas | todos |
| `performance_monitoring` | Monitoramento e métricas | todos |
| `experimental_features` | Features em desenvolvimento | dev apenas |

## 🔧 Funcionalidades Avançadas

### 🧠 Advanced Reasoning
```python
from utils.advanced_reasoning import reason_about

# Reasoning automático baseado na complexidade
trace = reason_about("Como implementar microserviços em Python?")

print(f"Resposta: {trace.final_answer}")
print(f"Complexidade: {trace.complexity.value}")
print(f"Steps executados: {len(trace.steps)}")
```

### 📊 Sistema de Evals
```python
from utils.prompt_evals import create_standard_eval_suite, run_quick_eval

# Avaliação rápida
results = run_quick_eval(
    prompts=["O que é Python?", "Como usar FastAPI?"],
    agent_function=my_agent
)
print(f"Score médio: {sum(results.values()) / len(results):.1f}%")

# Execução via linha de comando
python utils/eval_runner.py --agent local --suite reasoning --save
python utils/model_comparison.py --models local_agent,mock_agent --suite basic
```

### 🎛️ Feature Toggles
```python
from utils.feature_toggles import require_feature, is_feature_enabled

@require_feature("advanced_reasoning")
def complex_analysis(data):
    # Código que só executa se feature estiver habilitada
    return enhanced_process(data)

# Verificação manual
if is_feature_enabled("prompt_evaluation"):
    run_evaluation()
```

### 📝 Enhanced Logging
```python
from utils.enhanced_logging import get_enhanced_logger, TimingContext

logger = get_enhanced_logger("my_component")

# Log com contexto automático
logger.info("Processando request", user_id="123", operation="query")

# Timing automático
with TimingContext("database_query", logger):
    result = database.query(sql)  # Automaticamente medido e logado
```

## 🌐 Integração MCP

### MCP Tools
- `search_knowledge_base`: Busca na base vetorial ChromaDB
- `get_weather`: Informações meteorológicas atuais
- `langgraph_orchestrator`: Agente completo com orquestração inteligente
- `enhanced_reasoning`: Reasoning avançado multi-step
- `run_agent_evaluation`: Sistema de avaliações
- `manage_features`: Gerenciamento de feature toggles

### MCP Resources
- `config://agent`: Configuração do agente e ferramentas disponíveis
- `knowledge://base`: Informações sobre a base de conhecimento vetorial
- `status://system`: Status geral do sistema (servidor, agente, tools)
- `config://enhanced`: Configuração completa do servidor
- `metrics://system`: Métricas básicas do sistema

### MCP Prompts
- `agent-query`: Template flexível para consultas (estilos: conversational, technical, concise, educational)
- `knowledge-search`: Template otimizado para busca na base de conhecimento
- `weather-query`: Template para consultas meteorológicas
- `enhanced-agent-query`: Com reasoning opcional
- `evaluation-prompt`: Para cenários de teste

### Usando MCP Inspector
```bash
# Para servidor básico
fastmcp run mcp_server.py --transport streamable-http --host 127.0.0.1 --port 8088

# Para servidor enhanced (recomendado)
fastmcp run mcp_server_enhanced.py --transport streamable-http --host 127.0.0.1 --port 8089
```

Depois:
1. Execute o MCP Inspector: `mcp-inspector`
2. Abra o navegador em `http://localhost:3000`
3. Conecte em: `http://127.0.0.1:8089/mcp` (enhanced) ou `http://127.0.0.1:8088/mcp` (básico)
4. Explore Tools, Resources e Prompts na interface web

## 🛠️ Geração Automática de Tools

### Visão Geral
O sistema permite converter automaticamente especificações OpenAPI/Swagger em ferramentas LangGraph funcionais.

### Como Usar

#### 1. Preparar OpenAPI
Coloque sua especificação OpenAPI no arquivo `openapi/openapi.json`:

```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "Minha API",
    "version": "1.0.0"
  },
  "paths": {
    "/users": {
      "get": {
        "summary": "List Users",
        "operationId": "list_users"
      }
    }
  }
}
```

#### 2. Gerar Tools Automaticamente
```bash
python auto_tools.py
```

#### 3. Resultado
O script irá:
- ✅ Ler `openapi/openapi.json`
- ✅ Gerar `tools/api_tools_auto.py` com todas as ferramentas
- ✅ Atualizar automaticamente `tools/tools.py` para incluir as novas tools
- ✅ Integrar as ferramentas no agente sem necessidade de restart

## 🏗️ Arquitetura

### Componentes Principais

1. **Agent (agent.py)**
   - Interface de linha de comando
   - Loop interativo para conversação
   - Inicialização e configuração do sistema

2. **Graph (graphs/graph.py)**
   - Implementação do grafo LangGraph
   - Nós de decisão e execução
   - Estado do agente e fluxo de mensagens
   - Orquestração entre ferramentas e resposta direta

3. **Providers (providers/llm_providers.py)**
   - Sistema modular de provedores LLM
   - Suporte para Ollama, OpenAI, Gemini
   - Configuração automática via variáveis de ambiente

4. **Tools (tools/tools.py)**
   - Ferramentas LangChain (busca vetorial, clima)
   - Integração com ChromaDB e APIs externas

5. **Transform (transform/lang_mcp_transform.py)**
   - Bridge entre LangChain e MCP
   - Conversão automática de ferramentas
   - Wrapper assíncrono para compatibilidade

6. **MCP Servers**
   - **mcp_server.py**: Servidor MCP básico (desenvolvimento, testes, demos)
   - **mcp_server_enhanced.py**: Servidor completo com todas as funcionalidades (produção)

### Fluxo de Decisão

O agente decide automaticamente entre:

1. **Resposta Direta**: Para conversas gerais
2. **Busca na Base**: Para tópicos técnicos (python, IA, etc.)
3. **Ferramentas Externas**: Para clima, cálculos específicos
4. **Combinação**: Usar múltiplas ferramentas quando necessário

## 🚀 Deploy em Produção

### Docker
```bash
# Development
docker build --target development -t agent-core:dev .
docker run -p 8000:8000 --env-file .env agent-core:dev

# Production
docker build --target production -t agent-core:prod .
docker run -p 8000:8000 -e ENVIRONMENT=prod agent-core:prod

# Lambda
docker build --target lambda -t agent-core:lambda .
```

### AWS Lambda
```bash
# Build e deploy
docker build --target lambda -t agent-core:lambda .
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag agent-core:lambda <account>.dkr.ecr.<region>.amazonaws.com/agent-core:lambda
docker push <account>.dkr.ecr.<region>.amazonaws.com/agent-core:lambda

# Criar função Lambda
aws lambda create-function \
  --function-name agent-core \
  --code ImageUri=<account>.dkr.ecr.<region>.amazonaws.com/agent-core:lambda \
  --role arn:aws:iam::<account>:role/lambda-execution-role \
  --timeout 300 \
  --memory-size 1024
```

### Lambda Endpoints
- `GET /health`: Health check
- `POST /query`: Query principal ao agente
- `POST /evaluate`: Executa avaliações
- `GET /metrics`: Métricas do sistema
- `POST /features`: Gerencia feature toggles

## 🧪 Testes e Validação

### Testes de Integração
```bash
# Teste completo
python tests/integration_test.py

# Teste específico de MCP Enhanced
python tests/test_mcp_enhanced.py
```

### Health Checks
```bash
# Via MCP Server Enhanced
curl http://localhost:8000/health

# Via Lambda Handler
curl -X POST https://your-api.amazonaws.com/health
```

## 📊 Monitoramento

### Métricas Disponíveis
- **Feature Toggle Usage**: Quais features estão sendo usadas
- **Operation Timing**: Latência de operações críticas
- **Success/Error Rates**: Taxa de sucesso por operação
- **Reasoning Complexity**: Distribuição de complexidade das perguntas
- **Evaluation Scores**: Scores de qualidade dos prompts

### Logs Estruturados
```json
{
  "timestamp": "2024-01-15T10:30:45Z",
  "level": "INFO",
  "logger": "agent_core.reasoning",
  "message": "Reasoning concluído",
  "context": {
    "request_id": "req-123",
    "user_id": "user-456",
    "operation_name": "advanced_reasoning"
  },
  "duration_ms": 1250,
  "confidence": 0.85,
  "steps_executed": 4
}
```

## 🎯 Usar como Template

### Para Novo Projeto
1. Clone este repositório
2. Renomeie para seu projeto
3. Customize `feature_toggles.json` com suas features
4. Modifique `tools/tools.py` com suas tools específicas
5. Ajuste `providers/llm_providers.py` para seu provider
6. Configure `.env` para seu ambiente

### Features que Podem ser Removidas
- **OAuth2**: Remova se não precisar de autenticação
- **Prompt Evals**: Remova se não fizer avaliações sistemáticas
- **Advanced Reasoning**: Use reasoning básico se não precisar de multi-step
- **Lambda Handler**: Remova se não deployar em Lambda

### Customizações Comuns
```python
# Adicionar nova feature
# 1. Em feature_toggles.json
{
  "my_custom_feature": {
    "enabled": true,
    "description": "Minha feature customizada",
    "environments": ["dev", "prod"]
  }
}

# 2. No código
from utils.feature_toggles import require_feature

@require_feature("my_custom_feature")
def my_custom_function():
    return "Feature executada!"
```

## ⚠️ Considerações de Segurança

### OAuth2 em Produção
```bash
# Use AWS Secrets Manager
AWS_SECRET_NAME=agent-core-oauth2-secrets
AWS_SECRET_REGION=us-east-1

# Ou configure manualmente
OAUTH2_ISSUER=https://your-auth-server.com
OAUTH2_CLIENT_ID=your_client_id
OAUTH2_CLIENT_SECRET=your_client_secret
```

### Rate Limiting
```bash
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
```

### CORS em Produção
```bash
CORS_ORIGINS=https://yourdomain.com
CORS_METHODS=GET,POST
```

## 🐛 Troubleshooting

### Feature Toggle não funciona
```bash
# Verifique variáveis de ambiente
env | grep FEATURE_

# Teste manualmente
python -c "from utils.feature_toggles import is_feature_enabled; print(is_feature_enabled('advanced_reasoning'))"
```

### Enhanced Logging sem output
```bash
# Verifique nível de log
export LOG_LEVEL=DEBUG

# Verifique diretório logs
ls -la logs/
```

### OAuth2 falha na validação
```bash
# Teste conectividade JWKS
curl -v https://your-auth-server.com/.well-known/jwks.json

# Verifique configuração
python -c "from utils.oauth2_auth import OAuth2Config; print(OAuth2Config.from_env())"
```

## 📈 Exemplos de Uso

### Avaliação de Prompts
```bash
# Teste básico com agente mock
python utils/eval_runner.py --agent mock --suite basic

# Teste com agente local
python utils/eval_runner.py --agent local --suite reasoning --save

# Comparar modelos específicos
python utils/model_comparison.py --models local_agent,mock_agent --suite basic

# Gerar relatório HTML
python utils/eval_reporter.py --input results.json --format html
```

### Orquestração Inteligente
O agente possui um sistema de orquestração que decide automaticamente quando usar a base de conhecimento, ferramentas externas ou resposta direta baseado na análise da pergunta.

**Observação importante**: Se você encontrar erros com `search_knowledge_base`, verifique se o ChromaDB está configurado corretamente e se a base de conhecimento foi inicializada.

## 🤝 Contribuição

1. Fork o projeto
2. Crie branch para feature (`git checkout -b feature/AmazingFeature`)
3. Execute testes (`python tests/integration_test.py`)
4. Commit mudanças (`git commit -m 'Add AmazingFeature'`)
5. Push para branch (`git push origin feature/AmazingFeature`)
6. Abra Pull Request

## 📄 Licença

Distribuído sob licença MIT. Veja `LICENSE` para mais informações.

## 🎯 Roadmap

### Próximas Funcionalidades
- [ ] Integration com OpenTelemetry
- [ ] Support para Redis cache
- [ ] GraphQL API opcional
- [ ] Kubernetes Operators
- [ ] Multi-tenant support
- [ ] A/B testing framework integrado

### Melhorias Planejadas
- [ ] Performance optimizations
- [ ] Enhanced security features
- [ ] Better error handling
- [ ] Comprehensive documentation
- [ ] Video tutorials

---

**Agent Core Enhanced** - Template completo para agentes LangGraph empresariais 🚀