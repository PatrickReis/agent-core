"""
Testes de Integração Completos para Agent Core Enhanced.
Valida todas as funcionalidades implementadas de forma integrada.

Funcionalidades testadas:
- Feature Toggles e configurações
- Enhanced Logging e métricas
- OAuth2 Authentication (se habilitado)
- Advanced Reasoning multi-step
- Sistema de Evals
- MCP Server Enhanced
- Lambda Handler
- Integração entre componentes
"""

import os
import sys
import json
import time
import asyncio
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Adicionar o diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configurar ambiente de teste
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("FEATURE_ENHANCED_LOGGING_ENABLED", "true")
os.environ.setdefault("FEATURE_ADVANCED_REASONING_ENABLED", "true")
os.environ.setdefault("FEATURE_PROMPT_EVALUATION_ENABLED", "true")
os.environ.setdefault("FEATURE_OAUTH2_AUTHENTICATION_ENABLED", "false")  # Desabilitado para testes


class TestResult:
    """Resultado de um teste individual."""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = False
        self.error_message = None
        self.execution_time = 0.0
        self.details = {}
        self.start_time = time.time()

    def success(self, details: Dict[str, Any] = None):
        """Marca teste como bem-sucedido."""
        self.passed = True
        self.execution_time = time.time() - self.start_time
        self.details = details or {}

    def failure(self, error: str, details: Dict[str, Any] = None):
        """Marca teste como falhado."""
        self.passed = False
        self.error_message = error
        self.execution_time = time.time() - self.start_time
        self.details = details or {}


class IntegrationTestSuite:
    """Suite completa de testes de integração."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.setup_complete = False

    def log(self, message: str, level: str = "INFO"):
        """Log com timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def run_test(self, test_name: str, test_func):
        """Executa um teste individual."""
        result = TestResult(test_name)
        self.log(f"🧪 Executando: {test_name}")

        try:
            test_result = test_func()
            if asyncio.iscoroutine(test_result):
                test_result = asyncio.run(test_result)

            result.success(test_result)
            self.log(f"✅ {test_name} - {result.execution_time:.2f}s")

        except Exception as e:
            result.failure(str(e))
            self.log(f"❌ {test_name} - {str(e)}", "ERROR")
            self.log(traceback.format_exc(), "DEBUG")

        self.results.append(result)
        return result

    async def setup_test_environment(self):
        """Configura ambiente de teste."""
        self.log("🔧 Configurando ambiente de teste...")

        # Verificar se diretórios existem
        required_dirs = ["utils", "tools", "graphs", "providers", "logger"]
        for dir_name in required_dirs:
            if not Path(dir_name).exists():
                raise RuntimeError(f"Diretório requerido não encontrado: {dir_name}")

        self.setup_complete = True
        self.log("✅ Ambiente de teste configurado")

    def test_feature_toggles(self) -> Dict[str, Any]:
        """Teste do sistema de Feature Toggles."""
        try:
            from utils.feature_toggles import get_feature_manager, is_feature_enabled

            # Testar instanciação
            feature_manager = get_feature_manager()

            # Testar features básicas
            assert feature_manager is not None, "Feature manager não instanciado"

            # Testar verificação de features
            enhanced_logging_enabled = is_feature_enabled("enhanced_logging")
            assert enhanced_logging_enabled, "Enhanced logging deveria estar habilitado"

            # Testar status das features
            enabled_features = feature_manager.get_enabled_features()
            assert len(enabled_features) > 0, "Deveria haver features habilitadas"

            # Testar status detalhado
            status = feature_manager.get_feature_status()
            assert isinstance(status, dict), "Status deveria ser um dicionário"

            return {
                "enabled_features": enabled_features,
                "total_features": len(feature_manager.features),
                "status_keys": list(status.keys())
            }

        except ImportError as e:
            raise RuntimeError(f"Feature toggles não disponível: {e}")

    def test_enhanced_logging(self) -> Dict[str, Any]:
        """Teste do Enhanced Logging."""
        try:
            from utils.enhanced_logging import get_enhanced_logger, LogContext, TimingContext

            # Testar criação do logger
            logger = get_enhanced_logger("test_logger")
            assert logger is not None, "Logger não criado"

            # Testar logs básicos
            logger.info("Teste de log info")
            logger.debug("Teste de log debug")
            logger.warning("Teste de log warning")

            # Testar contexto
            LogContext.set_request_id("test-request-123")
            LogContext.set_operation_name("test-operation")

            assert LogContext.get_request_id() == "test-request-123"
            assert LogContext.get_operation_name() == "test-operation"

            # Testar timing context
            with TimingContext("test_timing", logger):
                time.sleep(0.01)  # Simular trabalho

            # Testar métricas se disponível
            metrics = {}
            if hasattr(logger, 'get_metrics'):
                metrics = logger.get_metrics()

            return {
                "logger_created": True,
                "context_working": True,
                "timing_working": True,
                "metrics_available": bool(metrics),
                "metrics_count": len(metrics) if metrics else 0
            }

        except ImportError as e:
            raise RuntimeError(f"Enhanced logging não disponível: {e}")

    def test_oauth2_auth(self) -> Dict[str, Any]:
        """Teste do sistema OAuth2 (se habilitado)."""
        try:
            from utils.oauth2_auth import OAuth2Config, TokenValidator, oauth2_enabled_check

            oauth2_enabled = oauth2_enabled_check()

            if not oauth2_enabled:
                return {
                    "oauth2_enabled": False,
                    "reason": "OAuth2 desabilitado via feature toggle"
                }

            # Testar configuração
            try:
                config = OAuth2Config.from_env()
                assert config is not None, "Configuração OAuth2 não criada"
            except Exception:
                # Configuração não está completa, o que é esperado em testes
                pass

            return {
                "oauth2_enabled": oauth2_enabled,
                "config_class_available": True
            }

        except ImportError as e:
            return {
                "oauth2_enabled": False,
                "error": f"OAuth2 não disponível: {e}"
            }

    def test_advanced_reasoning(self) -> Dict[str, Any]:
        """Teste do sistema de Advanced Reasoning."""
        try:
            from utils.advanced_reasoning import (
                ReasoningAnalyzer, AdvancedReasoningOrchestrator,
                get_reasoning_orchestrator, reason_about
            )

            # Testar analyzer
            analyzer = ReasoningAnalyzer()
            complexity = analyzer.analyze_complexity("O que é Python?")
            assert complexity is not None, "Análise de complexidade falhou"

            # Testar orquestrador
            orchestrator = AdvancedReasoningOrchestrator()
            assert orchestrator is not None, "Orquestrador não criado"

            # Testar reasoning simples (sem LLM para teste)
            test_question = "Teste de reasoning"
            trace = orchestrator.process_question(test_question)

            assert trace is not None, "Trace não criado"
            assert trace.question == test_question, "Pergunta não preservada"
            assert len(trace.steps) > 0, "Nenhum step executado"

            return {
                "analyzer_working": True,
                "orchestrator_working": True,
                "complexity_detected": complexity.value,
                "steps_executed": len(trace.steps),
                "reasoning_successful": trace.success
            }

        except ImportError as e:
            raise RuntimeError(f"Advanced reasoning não disponível: {e}")

    def test_prompt_evals(self) -> Dict[str, Any]:
        """Teste do sistema de Evals."""
        try:
            from utils.prompt_evals import (
                EvalFramework, RelevanceEvaluator, AccuracyEvaluator,
                create_standard_eval_suite, run_quick_eval
            )

            # Testar evaluators individuais
            relevance_eval = RelevanceEvaluator()
            result = relevance_eval.evaluate(
                prompt="O que é Python?",
                response="Python é uma linguagem de programação"
            )
            assert result.score >= 0, "Score de relevância inválido"

            # Testar framework
            framework = EvalFramework()
            assert framework is not None, "Framework não criado"

            # Testar avaliação rápida
            def mock_agent(prompt: str) -> str:
                return f"Resposta para: {prompt}"

            test_prompts = ["Teste 1", "Teste 2"]
            quick_results = run_quick_eval(test_prompts, mock_agent)

            return {
                "evaluators_working": True,
                "relevance_score": result.score,
                "framework_created": True,
                "quick_eval_results": len(quick_results),
                "eval_metrics": list(quick_results.keys()) if quick_results else []
            }

        except ImportError as e:
            raise RuntimeError(f"Prompt evals não disponível: {e}")

    def test_mcp_server_enhanced(self) -> Dict[str, Any]:
        """Teste do MCP Server Enhanced."""
        try:
            # Testar importação do servidor
            import mcp_server_enhanced

            # Verificar se servidor foi importado com sucesso
            assert hasattr(mcp_server_enhanced, 'mcp'), "Objeto MCP não encontrado"

            # Testar se funcionalidades aprimoradas estão disponíveis
            enhanced_available = getattr(mcp_server_enhanced, 'ENHANCED_FEATURES_AVAILABLE', False)

            # Testar se logger está funcionando
            logger = getattr(mcp_server_enhanced, 'logger', None)
            assert logger is not None, "Logger do MCP não inicializado"

            return {
                "server_imported": True,
                "enhanced_features_available": enhanced_available,
                "logger_initialized": True,
                "using_official_mcp": getattr(mcp_server_enhanced, 'using_official', False)
            }

        except ImportError as e:
            raise RuntimeError(f"MCP Server Enhanced não disponível: {e}")

    def test_lambda_handler(self) -> Dict[str, Any]:
        """Teste do Lambda Handler."""
        try:
            from lambda_handler import lambda_handler, LambdaRequest, LambdaResponse

            # Testar request wrapper
            mock_event = {
                "httpMethod": "GET",
                "path": "/health",
                "headers": {},
                "body": ""
            }

            class MockContext:
                aws_request_id = "test-123"
                function_name = "test-function"
                def get_remaining_time_in_millis(self):
                    return 300000

            # Testar handler principal
            response = lambda_handler(mock_event, MockContext())

            assert isinstance(response, dict), "Response não é um dicionário"
            assert "statusCode" in response, "StatusCode não encontrado"
            assert response["statusCode"] == 200, "Status code não é 200"

            # Testar query endpoint
            query_event = {
                "httpMethod": "POST",
                "path": "/query",
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"query": "Teste de query"})
            }

            query_response = lambda_handler(query_event, MockContext())
            assert query_response["statusCode"] in [200, 503], "Query response inválida"

            return {
                "handler_working": True,
                "health_check_status": response["statusCode"],
                "query_endpoint_status": query_response["statusCode"],
                "response_format_valid": True
            }

        except ImportError as e:
            raise RuntimeError(f"Lambda handler não disponível: {e}")

    def test_tools_integration(self) -> Dict[str, Any]:
        """Teste da integração com tools."""
        try:
            from tools.tools import tools

            assert len(tools) > 0, "Nenhuma tool carregada"

            # Testar se tools têm os atributos necessários
            for tool in tools:
                assert hasattr(tool, 'name'), f"Tool sem nome: {tool}"
                assert hasattr(tool, 'run'), f"Tool sem método run: {tool.name}"

            # Identificar tipos de tools disponíveis
            tool_names = [tool.name for tool in tools]

            return {
                "tools_loaded": len(tools),
                "tool_names": tool_names,
                "tools_valid": True
            }

        except ImportError as e:
            raise RuntimeError(f"Tools não disponíveis: {e}")

    def test_providers_integration(self) -> Dict[str, Any]:
        """Teste da integração com providers LLM."""
        try:
            from providers.llm_providers import get_llm, get_provider_info

            # Testar informações do provider
            provider_info = get_provider_info()
            assert isinstance(provider_info, dict), "Provider info não é um dicionário"

            # Testar se consegue obter LLM (pode falhar se não configurado)
            try:
                llm = get_llm()
                llm_available = llm is not None
            except Exception:
                llm_available = False

            return {
                "provider_info_available": True,
                "provider_name": provider_info.get("provider", "unknown"),
                "llm_available": llm_available,
                "model": provider_info.get("model", "unknown")
            }

        except ImportError as e:
            raise RuntimeError(f"Providers não disponíveis: {e}")

    def test_graph_integration(self) -> Dict[str, Any]:
        """Teste da integração com graphs."""
        try:
            from graphs.graph import create_agent_graph, AgentState

            # Testar se consegue importar
            assert create_agent_graph is not None, "Função create_agent_graph não disponível"
            assert AgentState is not None, "AgentState não disponível"

            # Testar criação do graph (pode falhar se LLM não disponível)
            try:
                from providers.llm_providers import get_llm
                llm = get_llm()
                graph = create_agent_graph(llm)
                graph_created = graph is not None
            except Exception:
                graph_created = False

            return {
                "graph_module_available": True,
                "agent_state_defined": True,
                "graph_creation_successful": graph_created
            }

        except ImportError as e:
            raise RuntimeError(f"Graph não disponível: {e}")

    async def run_all_tests(self):
        """Executa todos os testes."""
        self.log("🚀 Iniciando testes de integração do Agent Core Enhanced")

        # Setup
        await self.setup_test_environment()

        # Lista de testes a executar
        tests = [
            ("Feature Toggles", self.test_feature_toggles),
            ("Enhanced Logging", self.test_enhanced_logging),
            ("OAuth2 Authentication", self.test_oauth2_auth),
            ("Advanced Reasoning", self.test_advanced_reasoning),
            ("Prompt Evaluations", self.test_prompt_evals),
            ("MCP Server Enhanced", self.test_mcp_server_enhanced),
            ("Lambda Handler", self.test_lambda_handler),
            ("Tools Integration", self.test_tools_integration),
            ("Providers Integration", self.test_providers_integration),
            ("Graph Integration", self.test_graph_integration),
        ]

        # Executar testes
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)

    def generate_report(self) -> Dict[str, Any]:
        """Gera relatório dos testes."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests

        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "total_execution_time": sum(r.execution_time for r in self.results)
            },
            "tests": []
        }

        for result in self.results:
            test_data = {
                "name": result.test_name,
                "passed": result.passed,
                "execution_time": result.execution_time,
                "details": result.details
            }

            if not result.passed:
                test_data["error"] = result.error_message

            report["tests"].append(test_data)

        return report

    def print_report(self):
        """Imprime relatório formatado."""
        report = self.generate_report()

        self.log("=" * 60)
        self.log("📊 RELATÓRIO DE TESTES DE INTEGRAÇÃO")
        self.log("=" * 60)

        summary = report["summary"]
        self.log(f"Total de testes: {summary['total_tests']}")
        self.log(f"✅ Passou: {summary['passed']}")
        self.log(f"❌ Falhou: {summary['failed']}")
        self.log(f"📈 Taxa de sucesso: {summary['pass_rate']:.1f}%")
        self.log(f"⏱️ Tempo total: {summary['total_execution_time']:.2f}s")

        self.log("\n📋 DETALHES DOS TESTES:")
        self.log("-" * 60)

        for test in report["tests"]:
            status = "✅" if test["passed"] else "❌"
            self.log(f"{status} {test['name']} ({test['execution_time']:.2f}s)")

            if test["details"]:
                for key, value in test["details"].items():
                    self.log(f"   📌 {key}: {value}")

            if not test["passed"]:
                self.log(f"   🚨 Erro: {test['error']}")

            self.log("")

        # Salvar relatório em arquivo
        report_file = f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        self.log(f"📄 Relatório salvo em: {report_file}")

        return summary['pass_rate'] >= 70  # Considera sucesso se >= 70% dos testes passaram


async def main():
    """Função principal."""
    suite = IntegrationTestSuite()

    try:
        await suite.run_all_tests()
        success = suite.print_report()

        if success:
            suite.log("🎉 Testes de integração CONCLUÍDOS COM SUCESSO!")
            return 0
        else:
            suite.log("⚠️ Alguns testes falharam. Verifique o relatório para detalhes.")
            return 1

    except Exception as e:
        suite.log(f"❌ Erro crítico nos testes: {e}", "ERROR")
        suite.log(traceback.format_exc(), "DEBUG")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())