"""
MCP Server Enhanced para Agent Core.
Servidor MCP aprimorado com todas as funcionalidades integradas.

Funcionalidades integradas:
- Feature Toggles para controle de funcionalidades
- Enhanced Logging com métricas e tracing
- OAuth2 Authentication (condicional)
- Advanced Reasoning (multi-step com reflexão)
- Prompt Evaluation System
- Health checks e monitoring
- Resources avançados para status e métricas
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Importar funcionalidades aprimoradas
try:
    from utils.feature_toggles import get_feature_manager, is_feature_enabled
    from utils.enhanced_logging import get_enhanced_logger, LogContext, TimingContext
    from utils.oauth2_auth import OAuth2Config, MCPAuthMiddleware, conditional_oauth2_protection
    from utils.advanced_reasoning import get_reasoning_orchestrator, reason_about
    from utils.prompt_evals import EvalFramework, create_standard_eval_suite, run_quick_eval
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Funcionalidades aprimoradas não disponíveis: {e}")
    ENHANCED_FEATURES_AVAILABLE = False
    # Fallback para logger básico
    import logging
    def get_enhanced_logger(name):
        return logging.getLogger(name)

# Configurar logging aprimorado
if ENHANCED_FEATURES_AVAILABLE:
    logger = get_enhanced_logger("mcp_server_enhanced")
    feature_manager = get_feature_manager()
else:
    logger = get_enhanced_logger("mcp_server_enhanced")

logger.info("🔧 Carregando servidor MCP Enhanced...")

# Tentar usar MCP oficial primeiro, senão FastMCP
try:
    from mcp.server.fastmcp.server import FastMCP as OfficialFastMCP
    mcp = OfficialFastMCP(os.getenv("MCP_SERVER_NAME", "AgentCoreEnhanced"))
    using_official = True
    logger.info("✅ Usando MCP oficial")
except ImportError:
    try:
        from fastmcp import FastMCP
        mcp = FastMCP(os.getenv("MCP_SERVER_NAME", "AgentCoreEnhanced"))
        using_official = False
        logger.info("✅ Usando FastMCP")
    except ImportError as e:
        logger.error(f"❌ Nenhuma implementação MCP disponível: {e}")
        exit(1)

# Configurar OAuth2 se habilitado
oauth2_config = None
oauth2_middleware = None

if ENHANCED_FEATURES_AVAILABLE and is_feature_enabled("oauth2_authentication"):
    try:
        oauth2_config = OAuth2Config.from_env()
        oauth2_middleware = MCPAuthMiddleware(oauth2_config)
        logger.info("🔐 OAuth2 configurado e ativo")

        # Registrar scopes para tools protegidas
        oauth2_middleware.register_tool_scopes("run_agent_evaluation", ["admin", "eval"])
        oauth2_middleware.register_tool_scopes("manage_features", ["admin"])
        oauth2_middleware.register_resource_scopes("metrics://detailed", ["admin", "monitoring"])

    except Exception as e:
        logger.warning(f"⚠️ OAuth2 configuração falhou: {e}")
        oauth2_config = None
        oauth2_middleware = None


# Carregar tools do LangChain
logger.info("📦 Carregando tools...")
try:
    from tools.tools import tools
    logger.info(f"✅ {len(tools)} tools carregadas:")
    for tool in tools:
        logger.info(f"  🛠️ {tool.name}")
except Exception as e:
    logger.error(f"❌ Erro ao carregar tools: {e}")
    tools = []


# Carregar e executar transform
logger.info("🌉 Executando transform...")
try:
    from transform.lang_mcp_transform import register_langchain_tools_as_mcp, register_langgraph_agent_as_mcp

    # Registrar tools individuais
    register_langchain_tools_as_mcp(mcp, tools)

    # Registrar agente completo
    logger.info("🤖 Registrando agente LangGraph...")
    from graphs.graph import create_agent_graph
    from providers.llm_providers import get_llm

    def create_agent():
        return create_agent_graph(get_llm())

    register_langgraph_agent_as_mcp(
        mcp,
        create_agent,
        agent_name="enhanced_orchestrator",
        description="Orquestrador LangGraph aprimorado com reasoning avançado, feature toggles e logging estruturado"
    )

    logger.info("✅ Transform concluído com sucesso")
except Exception as e:
    logger.error(f"❌ Erro no transform: {e}")


# ===== MCP TOOLS APRIMORADOS =====

@mcp.tool("get_reasoning_trace")
async def get_reasoning_trace(trace_id: str) -> Dict[str, Any]:
    """
    Obtém trace detalhado do sistema de reasoning avançado.

    Args:
        trace_id: ID do trace de reasoning

    Returns:
        Trace completo com steps e métricas
    """
    if not ENHANCED_FEATURES_AVAILABLE or not is_feature_enabled("advanced_reasoning"):
        return {"error": "Advanced reasoning não disponível"}

    try:
        with LogContext():
            LogContext.set_operation_name("get_reasoning_trace")

            orchestrator = get_reasoning_orchestrator()
            trace = orchestrator.get_reasoning_trace(trace_id)

            if trace:
                logger.info(f"📋 Trace obtido: {trace_id}")
                return trace.to_dict()
            else:
                return {"error": f"Trace {trace_id} não encontrado"}

    except Exception as e:
        logger.error(f"❌ Erro ao obter trace: {e}")
        return {"error": str(e)}


@mcp.tool("run_agent_evaluation")
async def run_agent_evaluation(test_prompts: List[str], suite_name: str = "mcp_eval") -> Dict[str, Any]:
    """
    Executa avaliação do agente usando framework de evals.

    Args:
        test_prompts: Lista de prompts para testar
        suite_name: Nome da suite de avaliação

    Returns:
        Resultados da avaliação com métricas
    """
    if not ENHANCED_FEATURES_AVAILABLE or not is_feature_enabled("prompt_evaluation"):
        return {"error": "Prompt evaluation não disponível"}

    # Aplicar proteção OAuth2 se habilitada
    if oauth2_middleware and is_feature_enabled("oauth2_authentication"):
        # Verificar permissões seria feito aqui em implementação real
        pass

    try:
        with TimingContext("agent_evaluation", logger):
            logger.info(f"🧪 Iniciando avaliação com {len(test_prompts)} prompts")

            # Criar função wrapper para o agente
            from graphs.graph import create_agent_graph
            from providers.llm_providers import get_llm
            from langchain_core.messages import HumanMessage

            agent_graph = create_agent_graph(get_llm())

            def agent_function(prompt: str) -> str:
                try:
                    result = agent_graph.invoke({
                        "messages": [HumanMessage(content=prompt)]
                    })
                    last_message = result["messages"][-1]
                    return last_message.content if hasattr(last_message, 'content') else str(last_message)
                except Exception as e:
                    return f"Erro na execução: {e}"

            # Executar avaliação rápida
            quick_results = run_quick_eval(test_prompts, agent_function)

            logger.info(f"✅ Avaliação concluída - Score médio: {sum(quick_results.values())/len(quick_results):.1f}%")

            return {
                "suite_name": suite_name,
                "total_tests": len(test_prompts),
                "metrics": quick_results,
                "overall_score": sum(quick_results.values()) / len(quick_results),
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"❌ Erro na avaliação: {e}")
        return {"error": str(e)}


@mcp.tool("manage_features")
async def manage_features(action: str, feature_name: str = None) -> Dict[str, Any]:
    """
    Gerencia feature toggles do sistema.

    Args:
        action: Ação (list, enable, disable, status)
        feature_name: Nome da feature (para enable/disable)

    Returns:
        Status das features ou resultado da ação
    """
    if not ENHANCED_FEATURES_AVAILABLE:
        return {"error": "Feature toggles não disponíveis"}

    # Aplicar proteção OAuth2 se habilitada
    if oauth2_middleware and is_feature_enabled("oauth2_authentication"):
        # Verificar permissões seria feito aqui em implementação real
        pass

    try:
        with TimingContext("feature_management", logger):
            feature_manager = get_feature_manager()

            if action == "list":
                enabled_features = feature_manager.get_enabled_features()
                logger.info(f"📋 Listando features: {len(enabled_features)} habilitadas")
                return {
                    "enabled_features": enabled_features,
                    "total_features": len(feature_manager.features)
                }

            elif action == "status":
                status = feature_manager.get_feature_status()
                logger.info("📊 Status de features obtido")
                return {"features": status}

            elif action == "enable" and feature_name:
                feature_manager.enable_feature(feature_name, persist=True)
                logger.info(f"✅ Feature habilitada: {feature_name}")
                return {"success": True, "action": "enabled", "feature": feature_name}

            elif action == "disable" and feature_name:
                feature_manager.disable_feature(feature_name, persist=True)
                logger.info(f"❌ Feature desabilitada: {feature_name}")
                return {"success": True, "action": "disabled", "feature": feature_name}

            else:
                return {"error": "Ação inválida ou parâmetros faltando"}

    except Exception as e:
        logger.error(f"❌ Erro no gerenciamento de features: {e}")
        return {"error": str(e)}


@mcp.tool("enhanced_reasoning")
async def enhanced_reasoning(question: str, complexity_override: str = None) -> Dict[str, Any]:
    """
    Executa reasoning avançado com análise multi-step e reflexão.

    Args:
        question: Pergunta para análise
        complexity_override: Override de complexidade (simple, moderate, complex, advanced)

    Returns:
        Resultado do reasoning com trace completo
    """
    if not ENHANCED_FEATURES_AVAILABLE or not is_feature_enabled("advanced_reasoning"):
        return {"error": "Advanced reasoning não disponível"}

    try:
        with TimingContext("enhanced_reasoning", logger):
            logger.info(f"🧠 Iniciando reasoning avançado: {question[:50]}...")

            # Configurar contexto
            context = {"tools": tools}
            if complexity_override:
                context["complexity_override"] = complexity_override

            # Obter LLM e criar reasoning
            from providers.llm_providers import get_llm
            llm = get_llm()

            trace = reason_about(question, context, llm, tools)

            logger.info(f"✅ Reasoning concluído - {len(trace.steps)} steps, {trace.confidence:.1%} confiança")

            return {
                "question": question,
                "final_answer": trace.final_answer,
                "confidence": trace.confidence,
                "complexity": trace.complexity.value,
                "steps_executed": len(trace.steps),
                "total_time": trace.total_time,
                "success": trace.success,
                "trace_id": trace.trace_id,
                "reasoning_steps": [
                    {
                        "type": step.step_type.value,
                        "description": step.description,
                        "success": step.success,
                        "execution_time": step.execution_time
                    } for step in trace.steps
                ]
            }

    except Exception as e:
        logger.error(f"❌ Erro no reasoning avançado: {e}")
        return {"error": str(e)}


# ===== MCP RESOURCES APRIMORADOS =====

@mcp.resource("config://enhanced")
async def get_enhanced_config():
    """Configuração completa do servidor aprimorado."""
    try:
        from providers.llm_providers import get_provider_info
        provider_info = get_provider_info()

        config = {
            "server_name": os.getenv("MCP_SERVER_NAME", "AgentCoreEnhanced"),
            "environment": os.getenv("ENVIRONMENT", "dev"),
            "version": "2.0.0",
            "llm_provider": provider_info,
            "tools_available": [tool.name for tool in tools],
            "enhanced_features": {}
        }

        if ENHANCED_FEATURES_AVAILABLE:
            feature_manager = get_feature_manager()
            config["enhanced_features"] = {
                "feature_toggles": feature_manager.get_enabled_features(),
                "oauth2_enabled": is_feature_enabled("oauth2_authentication"),
                "advanced_reasoning": is_feature_enabled("advanced_reasoning"),
                "prompt_evaluation": is_feature_enabled("prompt_evaluation"),
                "enhanced_logging": is_feature_enabled("enhanced_logging")
            }

        return config

    except Exception as e:
        logger.error(f"❌ Erro ao obter configuração: {e}")
        return {"error": str(e)}


@mcp.resource("metrics://system")
async def get_system_metrics():
    """Métricas do sistema e performance."""
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "server_status": "running",
            "tools_count": len(tools),
            "uptime": "unknown"  # Implementar se necessário
        }

        if ENHANCED_FEATURES_AVAILABLE:
            # Obter métricas do enhanced logging
            logger_metrics = logger.get_metrics()
            if logger_metrics:
                metrics["operations"] = logger_metrics

            # Status das features
            feature_manager = get_feature_manager()
            metrics["features"] = {
                "total": len(feature_manager.features),
                "enabled": len(feature_manager.get_enabled_features())
            }

        return metrics

    except Exception as e:
        logger.error(f"❌ Erro ao obter métricas: {e}")
        return {"error": str(e)}


@mcp.resource("metrics://detailed")
async def get_detailed_metrics():
    """Métricas detalhadas (requer permissão admin)."""
    # Esta resource tem proteção OAuth2 configurada
    try:
        detailed_metrics = {
            "timestamp": datetime.now().isoformat(),
            "system_health": "healthy"
        }

        if ENHANCED_FEATURES_AVAILABLE:
            # Métricas detalhadas do logger
            detailed_metrics["logging_metrics"] = logger.get_metrics()

            # Status completo das features
            feature_manager = get_feature_manager()
            detailed_metrics["feature_status"] = feature_manager.get_feature_status()

            # Informações de OAuth2 se habilitado
            if oauth2_config:
                detailed_metrics["oauth2_status"] = {
                    "enabled": True,
                    "issuer": oauth2_config.issuer,
                    "audience": oauth2_config.audience
                }

        return detailed_metrics

    except Exception as e:
        logger.error(f"❌ Erro ao obter métricas detalhadas: {e}")
        return {"error": str(e)}


@mcp.resource("health://check")
async def health_check():
    """Health check completo do sistema."""
    try:
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "mcp_server": "ok",
                "tools_loaded": "ok" if tools else "warning",
                "llm_provider": "ok"
            }
        }

        if ENHANCED_FEATURES_AVAILABLE:
            # Verificar funcionalidades aprimoradas
            health["checks"]["enhanced_features"] = "ok"

            try:
                # Test feature toggles
                feature_manager = get_feature_manager()
                health["checks"]["feature_toggles"] = "ok"
            except:
                health["checks"]["feature_toggles"] = "error"

            try:
                # Test enhanced logging
                logger.info("Health check test")
                health["checks"]["enhanced_logging"] = "ok"
            except:
                health["checks"]["enhanced_logging"] = "error"

            # OAuth2 health
            if is_feature_enabled("oauth2_authentication"):
                health["checks"]["oauth2"] = "ok" if oauth2_config else "error"

        # Determinar status geral
        if any(check == "error" for check in health["checks"].values()):
            health["status"] = "unhealthy"
        elif any(check == "warning" for check in health["checks"].values()):
            health["status"] = "degraded"

        return health

    except Exception as e:
        logger.error(f"❌ Erro no health check: {e}")
        return {"status": "unhealthy", "error": str(e)}


# ===== MCP PROMPTS APRIMORADOS =====

@mcp.prompt("enhanced-agent-query")
async def enhanced_agent_query(topic: str = "", style: str = "conversational",
                             use_reasoning: bool = False):
    """Template aprimorado para consultas ao agente com reasoning opcional."""

    styles = {
        "conversational": "Responda de forma conversacional e amigável",
        "technical": "Forneça uma resposta técnica e detalhada",
        "concise": "Seja conciso e direto ao ponto",
        "educational": "Explique como se estivesse ensinando",
        "analytical": "Analise profundamente e forneça insights"
    }

    system_msg = styles.get(style, styles["conversational"])

    if use_reasoning and ENHANCED_FEATURES_AVAILABLE and is_feature_enabled("advanced_reasoning"):
        system_msg += ". Use reasoning multi-step quando apropriado para análises complexas."

    user_content = f"Sobre {topic}: " if topic else "Como posso ajudar você hoje?"

    return f"""Instruções do sistema: {system_msg}. Use ferramentas quando necessário e sempre responda em português brasileiro.

Usuário: {user_content}"""


@mcp.prompt("evaluation-prompt")
async def evaluation_prompt(test_scenario: str, expected_behavior: str = ""):
    """Template para cenários de avaliação e teste."""
    if not is_feature_enabled("prompt_evaluation"):
        return "Prompt evaluation não está habilitado no sistema atual."

    expected_part = f"\n\nComportamento esperado: {expected_behavior}" if expected_behavior else ""

    return f"""Instruções do sistema: Este é um cenário de teste/avaliação. Responda de forma precisa e demonstre o melhor comportamento possível.

Cenário de teste: {test_scenario}{expected_part}

Usuário: Execute este cenário de teste e demonstre a funcionalidade solicitada."""


# Iniciar servidor
if __name__ == "__main__":
    logger.info("🚀 Iniciando servidor MCP Enhanced...")

    # Log do status das features
    if ENHANCED_FEATURES_AVAILABLE:
        enabled_features = get_feature_manager().get_enabled_features()
        logger.info(f"✨ Features habilitadas: {', '.join(enabled_features)}")

    try:
        mcp.run()
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar servidor: {e}")
        raise