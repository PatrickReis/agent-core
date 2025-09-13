# Agent Core Enhanced 🚀

**Template completo para desenvolvimento de agentes LangGraph com funcionalidades empresariais avançadas.**

Este projeto serve como archetype/template base para criação de novos agentes LangGraph e servidores MCP com funcionalidades empresariais completas.

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

## 🏗️ Arquitetura

```
agent-core-enhanced/
├── utils/                     # Funcionalidades aprimoradas
│   ├── feature_toggles.py     # Sistema de feature toggles
│   ├── enhanced_logging.py    # Logging estruturado com métricas
│   ├── oauth2_auth.py         # Autenticação OAuth2 completa
│   ├── advanced_reasoning.py  # Reasoning multi-step
│   ├── prompt_evals.py        # Framework de avaliação
│   └── openapi_to_tools.py    # Utilitário migração tools
├── mcp_server_enhanced.py     # Servidor MCP com todas features
├── lambda_handler.py          # Handler otimizado para AWS Lambda
├── integration_test.py        # Testes completos de integração
├── feature_toggles.json       # Configuração de features
├── requirements-lambda.txt    # Deps específicas Lambda
├── Dockerfile                 # Multi-stage para todos ambientes
└── scripts/
    ├── setup-dev.sh           # Setup automático desenvolvimento
    └── test-features.sh       # Teste do sistema de features
```

## 🚀 Quick Start

### 1. Setup Automático
```bash
# Clone e configure automaticamente
git clone <repo>
cd agent-core-enhanced
chmod +x scripts/setup-dev.sh
./scripts/setup-dev.sh
```

### 2. Configuração Manual

```bash
# Instalar dependências
pip install -r requirements.txt

# Configurar ambiente
cp .env.example .env
# Edite .env com suas configurações

# Teste de integração
python integration_test.py
```

### 3. Executar

```bash
# Servidor MCP Enhanced
python mcp_server_enhanced.py

# Ou com MCP CLI
mcp dev mcp_server_enhanced.py

# Lambda local (opcional)
python lambda_handler.py
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

## 📋 Exemplos de Uso

### Feature Toggles
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

### Enhanced Logging
```python
from utils.enhanced_logging import get_enhanced_logger, TimingContext

logger = get_enhanced_logger("my_component")

# Log com contexto automático
logger.info("Processando request", user_id="123", operation="query")

# Timing automático
with TimingContext("database_query", logger):
    result = database.query(sql)  # Automaticamente medido e logado
```

### Advanced Reasoning
```python
from utils.advanced_reasoning import reason_about

# Reasoning automático baseado na complexidade
trace = reason_about("Como implementar microserviços em Python?")

print(f"Resposta: {trace.final_answer}")
print(f"Complexidade: {trace.complexity.value}")
print(f"Steps executados: {len(trace.steps)}")
```

### Prompt Evaluation
```python
from utils.prompt_evals import create_standard_eval_suite, run_quick_eval

# Avaliação rápida
results = run_quick_eval(
    prompts=["O que é Python?", "Como usar FastAPI?"],
    agent_function=my_agent
)
print(f"Score médio: {sum(results.values()) / len(results):.1f}%")
```

## 🔧 Deploy em Produção

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

### Kubernetes (exemplo)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-core
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-core
  template:
    metadata:
      labels:
        app: agent-core
    spec:
      containers:
      - name: agent-core
        image: agent-core:prod
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "prod"
        - name: FEATURE_OAUTH2_AUTHENTICATION_ENABLED
          value: "true"
```

## 🧪 Testes e Validação

### Testes de Integração
```bash
# Teste completo
python integration_test.py

# Teste específico de features
./scripts/test-features.sh

# Teste de performance
python -c "
from utils.enhanced_logging import get_enhanced_logger
from utils.feature_toggles import is_feature_enabled
import time

for i in range(1000):
    is_feature_enabled('advanced_reasoning')
"
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

## 📚 Documentação Avançada

### APIs Disponíveis

#### MCP Tools
- `enhanced_reasoning(question, complexity_override=None)`: Reasoning avançado
- `run_agent_evaluation(test_prompts, suite_name)`: Executa avaliações
- `manage_features(action, feature_name=None)`: Gerencia feature toggles
- `get_reasoning_trace(trace_id)`: Obtém trace de reasoning

#### MCP Resources
- `config://enhanced`: Configuração completa do servidor
- `metrics://system`: Métricas básicas do sistema
- `metrics://detailed`: Métricas detalhadas (requer permissão)
- `health://check`: Health check completo

#### Lambda Endpoints
- `GET /health`: Health check
- `POST /query`: Query principal ao agente
- `POST /evaluate`: Executa avaliações
- `GET /metrics`: Métricas do sistema
- `POST /features`: Gerencia feature toggles

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

## 🤝 Contribuição

1. Fork o projeto
2. Crie branch para feature (`git checkout -b feature/AmazingFeature`)
3. Execute testes (`python integration_test.py`)
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



  🏗️ Arquitetura do Sistema

  1. MCP Server Enhanced (mcp_server_enhanced.py)

  Como funciona como servidor MCP:
  - Usa FastMCP para criar servidor compatível com protocolo MCP
  - Registra tools, resources e prompts que clientes podem chamar
  - Servidor fica "ouvindo" e responde a chamadas via protocolo MCP
  - Clientes (como Claude Desktop) se conectam e podem executar as funcionalidades

  Estrutura principal:
  mcp_server_enhanced.py
  ├── Carrega dependências (tools, graphs, providers)
  ├── Configura features via feature_toggles
  ├── Registra tools MCP (@mcp.tool)
  ├── Registra resources MCP (@mcp.resource) 
  ├── Registra prompts MCP (@mcp.prompt)
  └── Inicia servidor (mcp.run())

  2. Sistema de Feature Toggles (utils/feature_toggles.py)

  Propósito: Controla quais funcionalidades estão ativas/inativas dinamicamente

  Funcionamento:
  - Arquivo JSON (feature_toggles.json) define o estado das features
  - Cada feature tem: enabled, environments, rollout_percentage, dependencies
  - Sistema verifica em runtime se feature está habilitada antes de executar
  - Permite ligar/desligar funcionalidades sem redeploy

  Features habilitadas no seu sistema:
  - prompt_evaluation - Sistema de evals
  - advanced_reasoning - Reasoning multi-step
  - enhanced_logging - Logging estruturado
  - performance_monitoring - Monitoramento

  3. Sistema de Avaliação (utils/prompt_evals.py)

  Como funciona o sistema de evals:

  Arquitetura do Eval Framework:
  EvalFramework
  ├── EvalSuiteBuilder (constrói suites de teste)
  ├── EvalSuite (executa testes)
  ├── BaseEvaluator (classe base para métricas)
  ├── Evaluators específicos (RelevanceEvaluator, AccuracyEvaluator, etc.)
  └── EvalResult (resultados com scores)

  Fluxo de avaliação:
  1. Criação: Builder cria suite com casos de teste + evaluators
  2. Execução: Suite executa cada teste no agente
  3. Avaliação: Cada evaluator analisa resposta e gera score
  4. Agregação: Calcula estatísticas finais (média, mediana, taxa de aprovação)
  5. Relatório: Gera relatório completo com métricas

  Evaluators implementados:
  - RelevanceEvaluator: Analisa keywords entre prompt/resposta
  - AccuracyEvaluator: Compara com resposta esperada
  - LatencyEvaluator: Mede tempo de resposta vs target
  - SafetyEvaluator: Detecta padrões inseguros via regex
  - ToolUsageEvaluator: Verifica uso apropriado de tools

  4. Como Funciona como Agente

  Integração LangChain/LangGraph:
  - transform/lang_mcp_transform.py converte tools LangChain para MCP
  - graphs/graph.py define o grafo do agente (estados, transições)
  - tools/tools.py contém ferramentas (weather, knowledge base, etc.)
  - providers/llm_providers.py abstrai diferentes LLMs

  Fluxo do agente:
  1. Entrada: Recebe query via MCP
  2. Processamento: LangGraph executa states/nodes do grafo
  3. Reasoning: Sistema de reasoning avançado (se habilitado)
  4. Tools: Chama ferramentas quando necessário
  5. Resposta: Retorna resultado estruturado via MCP

  5. Advanced Reasoning System

  Como funciona o reasoning avançado:
  - Analisa complexidade da pergunta (simple/moderate/complex/advanced)
  - Executa steps de reasoning em sequência
  - Cada step pode ser: analysis, tool_usage, reflection, synthesis
  - Gera trace completo com confiança e métricas
  - Permite obter traces detalhados posteriormente

  6. Enhanced Logging & Monitoring

  Sistema de observabilidade:
  - Logging estruturado com contexto e timing
  - Métricas de performance automáticas
  - Context tracing para debugar fluxos
  - Integração com feature toggles

  🔄 Fluxo Completo de Funcionamento

  Quando você executa uma avaliação:

  1. Cliente MCP chama tool run_agent_evaluation
  2. Servidor MCP valida se feature prompt_evaluation está ativa
  3. EvalFramework cria suite com evaluators padrão
  4. Para cada teste:
    - Executa agente LangGraph com o prompt
    - Mede tempo de execução
    - Cada evaluator analisa a resposta
    - Gera scores individuais
  5. Agrega resultados e calcula estatísticas finais
  6. Retorna relatório completo via MCP

  Integração entre componentes:
  - MCP Server = Interface de comunicação
  - Feature Toggles = Controle de funcionalidades
  - Eval System = Qualidade e métricas
  - Agent Graph = Lógica de processamento
  - Enhanced Logging = Observabilidade

  Essa arquitetura permite que o sistema funcione como um agente inteligente (via LangGraph) acessível via MCP (para clientes) com 
  sistema robusto de avaliação (para garantir qualidade) e controle dinâmico de features (para flexibilidade operacional).



  🛠️ Utilities Criadas:

  1. utils/eval_runner.py - Execução Individual

  # Teste básico com agente mock
  python utils/eval_runner.py --agent mock --suite basic

  # Teste com agente local
  python utils/eval_runner.py --agent local --suite reasoning --save

  # Avaliação rápida com prompts personalizados
  python utils/eval_runner.py --agent mock --quick "O que é Python?,Capital do Brasil?"

  # Usar prompts de arquivo
  python utils/eval_runner.py --agent local --custom-prompts my_prompts.txt

  2. utils/model_comparison.py - Comparação de Modelos

  # Comparar modelos específicos
  python utils/model_comparison.py --models local_agent,mock_agent --suite basic

  # Modo interativo
  python utils/model_comparison.py --interactive

  # Listar opções disponíveis
  python utils/model_comparison.py --list-models
  python utils/model_comparison.py --list-suites

  3. utils/eval_reporter.py - Geração de Relatórios

  # Gerar relatório HTML
  python utils/eval_reporter.py --input results.json --format html

  # Múltiplos formatos
  python utils/eval_reporter.py --input comparison.json --format all

  # Processar diretório inteiro
  python utils/eval_reporter.py --directory eval_results --format html

  4. config/eval_models.json - Configuração Central

  - Modelos disponíveis (GPT-4, Claude, Local Agent, Mock)
  - Suites de teste pré-configuradas
  - Configurações de evaluators
  - Presets de comparação

  🎯 Fluxo Completo de Uso:

  Teste Individual:

  python utils/eval_runner.py --agent local --suite comprehensive --save

  Comparação de Modelos:

  python utils/model_comparison.py --models local_agent,mock_agent --suite reasoning --save-report

  Geração de Relatório:

  python utils/eval_reporter.py --input comparison_results_*.json --format html

  📊 Recursos Incluídos:

  - 5 Suites de teste: basic, reasoning, technical, performance, tools_usage
  - 6 Modelos configurados: GPT-4, GPT-3.5, Claude Sonnet/Haiku, Local Agent, Mock
  - 5 Métricas automáticas: relevance, accuracy, latency, safety, tool_usage
  - 4 Formatos de relatório: HTML, JSON, CSV, Markdown
  - Modo interativo para seleção guiada
  - Configuração flexível via JSON

  Agora você pode executar testes completos de evals via linha de comando, comparar diferentes modelos e gerar
  relatórios profissionais automaticamente!
  