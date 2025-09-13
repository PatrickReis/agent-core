"""
Lambda Handler otimizado para Agent Core.
Handler para AWS Lambda com cold start mitigation e integração completa.

Funcionalidades:
- Cold start mitigation com inicialização global
- Validação de requests e error handling robusto
- Integração com todas as features aprimoradas
- Health checks e monitoring
- Context sharing entre invocações
"""

import json
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import traceback

# Configurar ambiente para Lambda
os.environ.setdefault("ENVIRONMENT", "lambda")

# Importações com fallback para funcionalidades aprimoradas
try:
    from utils.feature_toggles import get_feature_manager, is_feature_enabled
    from utils.enhanced_logging import get_enhanced_logger, LogContext, TimingContext
    from utils.oauth2_auth import OAuth2Config, MCPAuthMiddleware
    from utils.advanced_reasoning import get_reasoning_orchestrator
    from utils.prompt_evals import run_quick_eval
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced features not available in Lambda: {e}")
    ENHANCED_FEATURES_AVAILABLE = False
    import logging
    def get_enhanced_logger(name):
        return logging.getLogger(name)

# Configurar logging para Lambda
logger = get_enhanced_logger("lambda_handler")
logger.info("🚀 Inicializando Lambda Handler...")

# ===== INICIALIZAÇÃO GLOBAL (COLD START MITIGATION) =====

# Cache global para componentes pesados
_global_cache = {
    "llm_provider": None,
    "agent_graph": None,
    "tools": None,
    "feature_manager": None,
    "oauth2_middleware": None,
    "reasoning_orchestrator": None,
    "initialized": False,
    "initialization_time": None
}


def initialize_global_components():
    """Inicializa componentes globais para reduzir cold start."""
    if _global_cache["initialized"]:
        return

    start_time = time.time()
    logger.info("🔥 Inicializando componentes globais (cold start mitigation)...")

    try:
        # 1. Carregar Feature Manager
        if ENHANCED_FEATURES_AVAILABLE:
            _global_cache["feature_manager"] = get_feature_manager()
            logger.info("✅ Feature Manager carregado")

        # 2. Carregar Tools
        try:
            from tools.tools import tools
            _global_cache["tools"] = tools
            logger.info(f"✅ {len(tools)} tools carregadas")
        except Exception as e:
            logger.warning(f"⚠️ Tools não carregadas: {e}")
            _global_cache["tools"] = []

        # 3. Carregar LLM Provider
        try:
            from providers.llm_providers import get_llm
            _global_cache["llm_provider"] = get_llm()
            logger.info("✅ LLM Provider carregado")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar LLM: {e}")

        # 4. Criar Agent Graph
        if _global_cache["llm_provider"]:
            try:
                from graphs.graph import create_agent_graph
                _global_cache["agent_graph"] = create_agent_graph(_global_cache["llm_provider"])
                logger.info("✅ Agent Graph criado")
            except Exception as e:
                logger.error(f"❌ Erro ao criar Agent Graph: {e}")

        # 5. OAuth2 se habilitado
        if ENHANCED_FEATURES_AVAILABLE and is_feature_enabled("oauth2_authentication"):
            try:
                oauth2_config = OAuth2Config.from_env()
                _global_cache["oauth2_middleware"] = MCPAuthMiddleware(oauth2_config)
                logger.info("✅ OAuth2 Middleware configurado")
            except Exception as e:
                logger.warning(f"⚠️ OAuth2 não configurado: {e}")

        # 6. Reasoning Orchestrator
        if ENHANCED_FEATURES_AVAILABLE and is_feature_enabled("advanced_reasoning"):
            try:
                _global_cache["reasoning_orchestrator"] = get_reasoning_orchestrator(
                    _global_cache["llm_provider"],
                    _global_cache["tools"]
                )
                logger.info("✅ Reasoning Orchestrator carregado")
            except Exception as e:
                logger.warning(f"⚠️ Reasoning Orchestrator não carregado: {e}")

        _global_cache["initialized"] = True
        _global_cache["initialization_time"] = time.time() - start_time

        logger.info(f"🎯 Inicialização completa em {_global_cache['initialization_time']:.2f}s")

    except Exception as e:
        logger.error(f"❌ Erro na inicialização global: {e}")
        # Continuar mesmo com erro para não quebrar completamente


# Inicializar componentes na importação do módulo
initialize_global_components()


# ===== CLASSES DE SUPORTE =====

class LambdaRequest:
    """Encapsula request do Lambda."""

    def __init__(self, event: Dict[str, Any], context: Any):
        self.event = event
        self.context = context
        self.request_id = getattr(context, 'aws_request_id', 'unknown')
        self.function_name = getattr(context, 'function_name', 'agent-core-lambda')
        self.remaining_time = getattr(context, 'get_remaining_time_in_millis', lambda: 30000)

    def get_body(self) -> Dict[str, Any]:
        """Extrai body do request."""
        body = self.event.get('body', '{}')
        if isinstance(body, str):
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {}
        return body if isinstance(body, dict) else {}

    def get_headers(self) -> Dict[str, str]:
        """Extrai headers do request."""
        return self.event.get('headers', {})

    def get_query_params(self) -> Dict[str, str]:
        """Extrai query parameters."""
        return self.event.get('queryStringParameters', {}) or {}

    def get_path(self) -> str:
        """Extrai path do request."""
        return self.event.get('path', '/')

    def get_method(self) -> str:
        """Extrai HTTP method."""
        return self.event.get('httpMethod', 'POST')


class LambdaResponse:
    """Encapsula response do Lambda."""

    def __init__(self, status_code: int = 200, body: Any = None,
                 headers: Dict[str, str] = None, cors: bool = True):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}

        # Adicionar CORS headers se solicitado
        if cors:
            self.headers.update({
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With"
            })

    def to_dict(self) -> Dict[str, Any]:
        """Converte para formato Lambda response."""
        body_str = json.dumps(self.body, ensure_ascii=False, default=str) if self.body else "{}"

        return {
            "statusCode": self.status_code,
            "headers": self.headers,
            "body": body_str
        }


# ===== HANDLERS ESPECÍFICOS =====

def handle_agent_query(request: LambdaRequest) -> LambdaResponse:
    """Handler para consultas ao agente."""
    try:
        body = request.get_body()
        query = body.get("query") or body.get("message", "")

        if not query:
            return LambdaResponse(400, {"error": "Query não fornecida"})

        # Configurar contexto de logging
        if ENHANCED_FEATURES_AVAILABLE:
            LogContext.set_request_id(request.request_id)
            LogContext.set_operation_name("agent_query")

        # Usar reasoning avançado se disponível
        if (ENHANCED_FEATURES_AVAILABLE and
            is_feature_enabled("advanced_reasoning") and
            _global_cache["reasoning_orchestrator"]):

            logger.info(f"🧠 Usando reasoning avançado para: {query[:50]}...")

            with TimingContext("advanced_reasoning", logger):
                trace = _global_cache["reasoning_orchestrator"].process_question(query)

                response_data = {
                    "answer": trace.final_answer,
                    "confidence": trace.confidence,
                    "reasoning_used": True,
                    "complexity": trace.complexity.value,
                    "steps_executed": len(trace.steps),
                    "execution_time": trace.total_time,
                    "trace_id": trace.trace_id
                }

                return LambdaResponse(200, response_data)

        # Fallback para agente básico
        elif _global_cache["agent_graph"]:
            logger.info(f"🤖 Usando agente básico para: {query[:50]}...")

            from langchain_core.messages import HumanMessage

            with TimingContext("basic_agent", logger):
                result = _global_cache["agent_graph"].invoke({
                    "messages": [HumanMessage(content=query)]
                })

                last_message = result["messages"][-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

                response_data = {
                    "answer": answer,
                    "reasoning_used": False,
                    "method": "basic_agent"
                }

                return LambdaResponse(200, response_data)

        else:
            return LambdaResponse(503, {"error": "Agente não disponível"})

    except Exception as e:
        logger.error(f"❌ Erro no agent query: {e}")
        return LambdaResponse(500, {"error": str(e)})


def handle_evaluation(request: LambdaRequest) -> LambdaResponse:
    """Handler para avaliações."""
    if not ENHANCED_FEATURES_AVAILABLE or not is_feature_enabled("prompt_evaluation"):
        return LambdaResponse(503, {"error": "Prompt evaluation não disponível"})

    try:
        body = request.get_body()
        test_prompts = body.get("test_prompts", [])

        if not test_prompts:
            return LambdaResponse(400, {"error": "test_prompts não fornecidos"})

        logger.info(f"🧪 Iniciando avaliação com {len(test_prompts)} prompts")

        # Função wrapper para o agente
        def agent_function(prompt: str) -> str:
            if _global_cache["agent_graph"]:
                from langchain_core.messages import HumanMessage
                result = _global_cache["agent_graph"].invoke({
                    "messages": [HumanMessage(content=prompt)]
                })
                last_message = result["messages"][-1]
                return last_message.content if hasattr(last_message, 'content') else str(last_message)
            return "Agente não disponível"

        # Executar avaliação
        with TimingContext("evaluation", logger):
            results = run_quick_eval(test_prompts, agent_function)

            response_data = {
                "evaluation_results": results,
                "overall_score": sum(results.values()) / len(results) if results else 0,
                "total_tests": len(test_prompts),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            return LambdaResponse(200, response_data)

    except Exception as e:
        logger.error(f"❌ Erro na avaliação: {e}")
        return LambdaResponse(500, {"error": str(e)})


def handle_health_check(request: LambdaRequest) -> LambdaResponse:
    """Handler para health check."""
    try:
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "function_name": request.function_name,
            "request_id": request.request_id,
            "remaining_time_ms": request.remaining_time(),
            "cold_start": not _global_cache["initialized"],
            "initialization_time": _global_cache.get("initialization_time", 0),
            "components": {
                "llm_provider": "ok" if _global_cache["llm_provider"] else "error",
                "agent_graph": "ok" if _global_cache["agent_graph"] else "error",
                "tools": f"{len(_global_cache['tools'] or [])} loaded"
            }
        }

        if ENHANCED_FEATURES_AVAILABLE:
            feature_manager = _global_cache["feature_manager"]
            if feature_manager:
                enabled_features = feature_manager.get_enabled_features()
                health_data["features"] = {
                    "enabled": enabled_features,
                    "total": len(feature_manager.features)
                }

        # Determinar status geral
        if not _global_cache["llm_provider"] or not _global_cache["agent_graph"]:
            health_data["status"] = "degraded"

        return LambdaResponse(200, health_data)

    except Exception as e:
        logger.error(f"❌ Erro no health check: {e}")
        return LambdaResponse(500, {"error": str(e)})


def handle_metrics(request: LambdaRequest) -> LambdaResponse:
    """Handler para métricas."""
    try:
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "function_name": request.function_name,
            "initialization_time": _global_cache.get("initialization_time", 0),
            "components_loaded": {
                "llm_provider": bool(_global_cache["llm_provider"]),
                "agent_graph": bool(_global_cache["agent_graph"]),
                "tools_count": len(_global_cache["tools"] or [])
            }
        }

        if ENHANCED_FEATURES_AVAILABLE and logger.metrics_collector:
            metrics["operations"] = logger.get_metrics()

        return LambdaResponse(200, metrics)

    except Exception as e:
        logger.error(f"❌ Erro ao obter métricas: {e}")
        return LambdaResponse(500, {"error": str(e)})


def handle_feature_management(request: LambdaRequest) -> LambdaResponse:
    """Handler para gerenciamento de features."""
    if not ENHANCED_FEATURES_AVAILABLE:
        return LambdaResponse(503, {"error": "Feature toggles não disponíveis"})

    try:
        body = request.get_body()
        action = body.get("action", "list")

        feature_manager = _global_cache["feature_manager"]
        if not feature_manager:
            return LambdaResponse(503, {"error": "Feature manager não disponível"})

        if action == "list":
            return LambdaResponse(200, {
                "enabled_features": feature_manager.get_enabled_features(),
                "total_features": len(feature_manager.features)
            })
        elif action == "status":
            return LambdaResponse(200, {
                "features": feature_manager.get_feature_status()
            })
        else:
            return LambdaResponse(400, {"error": "Ação não suportada no Lambda"})

    except Exception as e:
        logger.error(f"❌ Erro no gerenciamento de features: {e}")
        return LambdaResponse(500, {"error": str(e)})


# ===== ROUTER PRINCIPAL =====

def route_request(request: LambdaRequest) -> LambdaResponse:
    """Roteador principal para diferentes tipos de request."""
    path = request.get_path()
    method = request.get_method()

    # OPTIONS request (CORS preflight)
    if method == "OPTIONS":
        return LambdaResponse(200, {}, cors=True)

    # Health check
    if path == "/health" or path == "/":
        return handle_health_check(request)

    # Métricas
    elif path == "/metrics":
        return handle_metrics(request)

    # Agent query (endpoint principal)
    elif path == "/query" or path == "/agent":
        return handle_agent_query(request)

    # Avaliação
    elif path == "/evaluate":
        return handle_evaluation(request)

    # Gerenciamento de features
    elif path == "/features":
        return handle_feature_management(request)

    # Rota não encontrada
    else:
        return LambdaResponse(404, {"error": f"Rota não encontrada: {path}"})


# ===== HANDLER PRINCIPAL =====

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handler principal do AWS Lambda.

    Args:
        event: Evento Lambda (API Gateway, ALB, etc.)
        context: Contexto Lambda

    Returns:
        Response formatado para Lambda
    """
    request_start = time.time()

    try:
        # Criar request wrapper
        request = LambdaRequest(event, context)

        # Logging inicial
        logger.info(f"📨 Request recebido: {request.get_method()} {request.get_path()}")

        # Verificar tempo restante
        if request.remaining_time() < 5000:  # Menos de 5 segundos
            logger.warning("⚠️ Pouco tempo restante, processamento pode ser interrompido")

        # Configurar contexto se enhanced features disponíveis
        if ENHANCED_FEATURES_AVAILABLE:
            LogContext.set_request_id(request.request_id)

        # Processar request
        response = route_request(request)

        # Log de sucesso
        execution_time = time.time() - request_start
        logger.info(f"✅ Request processado em {execution_time:.2f}s - Status: {response.status_code}")

        return response.to_dict()

    except Exception as e:
        # Error handling robusto
        execution_time = time.time() - request_start
        logger.error(f"❌ Erro não tratado: {e}")
        logger.error(traceback.format_exc())

        error_response = LambdaResponse(500, {
            "error": "Internal server error",
            "message": str(e),
            "execution_time": execution_time,
            "request_id": getattr(context, 'aws_request_id', 'unknown')
        })

        return error_response.to_dict()


# ===== HANDLER PARA MANGUM (OPCIONAL) =====

# Para usar com frameworks ASGI como FastAPI via Mangum
try:
    from mangum import Mangum

    # Criar app ASGI simples se necessário
    def create_asgi_app():
        """Cria aplicação ASGI para uso com Mangum."""
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse

        app = FastAPI(title="Agent Core Lambda API")

        @app.get("/health")
        async def health():
            return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

        @app.post("/query")
        async def query(request_data: Dict[str, Any]):
            # Simular event/context do Lambda
            event = {
                "body": json.dumps(request_data),
                "path": "/query",
                "httpMethod": "POST"
            }

            class MockContext:
                aws_request_id = "asgi-request"
                function_name = "agent-core-asgi"
                def get_remaining_time_in_millis(self):
                    return 300000

            response = lambda_handler(event, MockContext())
            return JSONResponse(json.loads(response["body"]), status_code=response["statusCode"])

        return app

    # Handler para Mangum
    asgi_app = create_asgi_app()
    handler = Mangum(asgi_app)

except ImportError:
    handler = lambda_handler  # Usar handler padrão se Mangum não disponível


if __name__ == "__main__":
    # Teste local
    print("🧪 Testando Lambda Handler localmente...")

    test_event = {
        "httpMethod": "GET",
        "path": "/health",
        "headers": {},
        "body": ""
    }

    class TestContext:
        aws_request_id = "test-request"
        function_name = "test-function"
        def get_remaining_time_in_millis(self):
            return 300000

    response = lambda_handler(test_event, TestContext())
    print(f"📋 Response: {json.dumps(response, indent=2)}")

    # Teste de query
    query_event = {
        "httpMethod": "POST",
        "path": "/query",
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"query": "O que é Python?"})
    }

    response = lambda_handler(query_event, TestContext())
    print(f"🤖 Query Response: {json.dumps(response, indent=2)}")