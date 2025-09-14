#!/usr/bin/env python3
"""
Utility para Comparação de Modelos via Evals.
Permite comparar diferentes modelos LLM usando o mesmo conjunto de testes.

Usage:
    python utils/model_comparison.py --models gpt4,claude --suite basic
    python utils/model_comparison.py --config comparison_config.json
    python utils/model_comparison.py --interactive
"""

import os
import sys
import json
import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, asdict

# Adicionar o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.prompt_evals import EvalFramework, create_standard_eval_suite, EvalSuiteResult
    from utils.enhanced_logging import get_enhanced_logger
    from utils.feature_toggles import is_feature_enabled
    EVALS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Sistema de evals não disponível: {e}")
    EVALS_AVAILABLE = False

logger = get_enhanced_logger("model_comparison") if EVALS_AVAILABLE else None


@dataclass
class ModelConfig:
    """Configuração de um modelo para comparação."""
    name: str
    description: str
    provider: str  # openai, anthropic, local, etc
    model_id: str  # gpt-4, claude-3, llama-2, etc
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    parameters: Dict[str, Any] = None
    enabled: bool = True

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class ComparisonResult:
    """Resultado de comparação entre modelos."""
    comparison_id: str
    timestamp: str
    suite_name: str
    models: List[str]
    individual_results: Dict[str, EvalSuiteResult]
    comparison_metrics: Dict[str, Any]
    winner: Optional[str] = None
    summary: Dict[str, Any] = None

    def save_to_file(self, file_path: Optional[str] = None):
        """Salva resultado de comparação em arquivo."""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"comparison_results_{self.comparison_id}_{timestamp}.json"

        # Converter para formato serializável
        data = asdict(self)

        # Converter EvalSuiteResult para dict
        for model_name, result in self.individual_results.items():
            if hasattr(result, 'to_dict'):
                data['individual_results'][model_name] = result.to_dict()
            else:
                data['individual_results'][model_name] = str(result)

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        print(f"📁 Resultado salvo em: {file_path}")
        return file_path


class ModelComparisonFramework:
    """Framework para comparação de modelos."""

    def __init__(self):
        self.eval_framework = EvalFramework() if EVALS_AVAILABLE else None
        self.models: Dict[str, ModelConfig] = {}
        self.test_suites: Dict[str, List[Dict[str, Any]]] = {}
        self._load_models_from_config()
        self._load_default_test_suites()

    def _load_default_models(self):
        """Carrega configurações padrão de modelos."""
        default_models = [
            ModelConfig(
                name="gpt4",
                description="OpenAI GPT-4",
                provider="openai",
                model_id="gpt-4",
                api_key_env="OPENAI_API_KEY",
                parameters={"temperature": 0.1, "max_tokens": 1000}
            ),
            ModelConfig(
                name="gpt3.5",
                description="OpenAI GPT-3.5 Turbo",
                provider="openai",
                model_id="gpt-3.5-turbo",
                api_key_env="OPENAI_API_KEY",
                parameters={"temperature": 0.1, "max_tokens": 1000}
            ),
            ModelConfig(
                name="claude",
                description="Anthropic Claude",
                provider="anthropic",
                model_id="claude-3-sonnet-20240229",
                api_key_env="ANTHROPIC_API_KEY",
                parameters={"temperature": 0.1, "max_tokens": 1000}
            ),
            ModelConfig(
                name="local_agent",
                description="Agent Core Local (via MCP)",
                provider="local",
                model_id="agent_core_enhanced",
                enabled=True
            )
        ]

        for model in default_models:
            self.models[model.name] = model

    def _load_models_from_config(self):
        """Carrega modelos do arquivo de configuração JSON."""
        # Primeiro carrega os modelos padrão como fallback
        self._load_default_models()

        # Tentar carregar do arquivo de configuração
        config_file = Path("config/eval_models.json")
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                models_config = config_data.get("models", {})
                for model_name, model_data in models_config.items():
                    self.models[model_name] = ModelConfig(
                        name=model_name,
                        description=model_data.get("description", ""),
                        provider=model_data.get("provider", ""),
                        model_id=model_data.get("model_id", ""),
                        api_key_env=model_data.get("api_key_env"),
                        base_url=model_data.get("base_url"),
                        parameters=model_data.get("parameters", {}),
                        enabled=model_data.get("enabled", True)
                    )

                print(f"✅ Carregados {len(models_config)} modelos do arquivo de configuração")

            except Exception as e:
                print(f"⚠️ Erro ao carregar config/eval_models.json: {e}")
                print("Usando apenas modelos padrão")
        else:
            print("⚠️ Arquivo config/eval_models.json não encontrado")
            print("Usando apenas modelos padrão")

    def _load_default_test_suites(self):
        """Carrega suites de teste padrão."""
        self.test_suites = {
            "basic": [
                {"prompt": "O que é Python?", "expected": "Python é uma linguagem de programação"},
                {"prompt": "Qual a capital do Brasil?", "expected": "Brasília"},
                {"prompt": "Como calcular a área de um círculo?", "tags": ["matemática"]}
            ],

            "reasoning": [
                {"prompt": "Se todos os gatos são animais e alguns animais voam, alguns gatos voam?"},
                {"prompt": "Por que o céu é azul? Explique de forma simples."},
                {"prompt": "Compare os prós e contras de energia solar vs nuclear"}
            ],

            "coding": [
                {"prompt": "Escreva uma função Python para calcular fibonacci"},
                {"prompt": "Explique o conceito de recursão com exemplo"},
                {"prompt": "Como otimizar uma consulta SQL lenta?", "tags": ["sql", "performance"]}
            ],

            "knowledge": [
                {"prompt": "Explique machine learning em 2 frases"},
                {"prompt": "Qual a diferença entre HTTP e HTTPS?"},
                {"prompt": "Como funciona a internet? Resposta técnica.", "tags": ["networking"]}
            ],

            "tools": [
                {"prompt": "Qual a temperatura em São Paulo hoje?", "tags": ["weather", "tool_usage"]},
                {"prompt": "Procure informações sobre inteligência artificial", "tags": ["search", "tool_usage"]},
                {"prompt": "Como está o clima em Londres?", "tags": ["weather", "tool_usage"]}
            ]
        }

    def add_model(self, model_config: ModelConfig):
        """Adiciona modelo para comparação."""
        self.models[model_config.name] = model_config
        print(f"➕ Modelo adicionado: {model_config.name} ({model_config.description})")

    def add_test_suite(self, suite_name: str, test_cases: List[Dict[str, Any]]):
        """Adiciona suite de teste personalizada."""
        self.test_suites[suite_name] = test_cases
        print(f"📝 Suite adicionada: {suite_name} ({len(test_cases)} testes)")

    def create_model_function(self, model_name: str) -> Callable[[str], str]:
        """Cria função wrapper para um modelo específico."""
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} não encontrado")

        model_config = self.models[model_name]

        def model_function(prompt: str) -> str:
            """Função que executa prompt no modelo específico."""
            try:
                if model_config.provider == "local":
                    # Usar agente local via import
                    return self._call_local_agent(prompt)
                elif model_config.provider == "openai":
                    return self._call_openai_model(prompt, model_config)
                elif model_config.provider == "anthropic":
                    return self._call_anthropic_model(prompt, model_config)
                elif model_config.provider == "ollama":
                    return self._call_ollama_model(prompt, model_config)
                elif model_config.provider == "gemini":
                    return self._call_gemini_model(prompt, model_config)
                else:
                    return f"Modelo {model_name} não implementado ainda"
            except Exception as e:
                return f"Erro ao executar {model_name}: {str(e)}"

        return model_function

    def _call_local_agent(self, prompt: str) -> str:
        """Chama agente local do Agent Core."""
        try:
            # Tentar usar o graph do agente
            from graphs.graph import create_agent_graph
            from providers.llm_providers import get_llm
            from langchain_core.messages import HumanMessage

            agent_graph = create_agent_graph(get_llm())
            result = agent_graph.invoke({
                "messages": [HumanMessage(content=prompt)]
            })

            last_message = result["messages"][-1]
            return last_message.content if hasattr(last_message, 'content') else str(last_message)
        except Exception as e:
            return f"Erro no agente local: {e}"

    def _call_openai_model(self, prompt: str, config: ModelConfig) -> str:
        """Chama modelo OpenAI."""
        try:
            import openai
            api_key = os.getenv(config.api_key_env) if config.api_key_env else None

            if not api_key:
                return f"API key não encontrada para OpenAI ({config.api_key_env})"

            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=config.model_id,
                messages=[{"role": "user", "content": prompt}],
                **config.parameters
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"Erro OpenAI: {e}"

    def _call_anthropic_model(self, prompt: str, config: ModelConfig) -> str:
        """Chama modelo Anthropic."""
        try:
            import anthropic
            api_key = os.getenv(config.api_key_env) if config.api_key_env else None

            if not api_key:
                return f"API key não encontrada para Anthropic ({config.api_key_env})"

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=config.model_id,
                messages=[{"role": "user", "content": prompt}],
                **config.parameters
            )

            return response.content[0].text
        except Exception as e:
            return f"Erro Anthropic: {e}"

    def _call_ollama_model(self, prompt: str, config: ModelConfig) -> str:
        """Chama modelo via Ollama."""
        try:
            import requests

            base_url = config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

            # Endpoint do Ollama
            url = f"{base_url}/api/generate"

            payload = {
                "model": config.model_id,
                "prompt": prompt,
                "stream": False,
                **config.parameters
            }

            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            return result.get("response", "Erro: resposta vazia do Ollama")

        except Exception as e:
            return f"Erro Ollama: {e}"

    def _call_gemini_model(self, prompt: str, config: ModelConfig) -> str:
        """Chama modelo Google Gemini."""
        try:
            import google.generativeai as genai

            api_key = os.getenv(config.api_key_env) if config.api_key_env else None
            if not api_key:
                return f"API key não encontrada para Gemini ({config.api_key_env})"

            # Configurar Gemini
            genai.configure(api_key=api_key)

            # Criar modelo
            model = genai.GenerativeModel(config.model_id)

            # Parâmetros de geração
            generation_config = {
                "temperature": config.parameters.get("temperature", 0.7),
                "max_output_tokens": config.parameters.get("max_tokens", 2048),
            }

            # Gerar resposta
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )

            return response.text

        except Exception as e:
            return f"Erro Gemini: {e}"

    def compare_models(self, model_names: List[str], suite_name: str) -> ComparisonResult:
        """Compara múltiplos modelos usando uma suite de testes."""
        if not EVALS_AVAILABLE:
            raise ValueError("Sistema de evals não disponível")

        if suite_name not in self.test_suites:
            raise ValueError(f"Suite {suite_name} não encontrada")

        print(f"\n🏁 Iniciando comparação de modelos")
        print(f"📊 Modelos: {', '.join(model_names)}")
        print(f"📝 Suite: {suite_name} ({len(self.test_suites[suite_name])} testes)")

        # Criar suite de avaliação
        test_cases = self.test_suites[suite_name]
        eval_suite = create_standard_eval_suite(f"comparison_{suite_name}", test_cases)

        # Executar cada modelo
        individual_results = {}
        for model_name in model_names:
            print(f"\n🤖 Testando modelo: {model_name}")

            if model_name not in self.models:
                print(f"❌ Modelo {model_name} não encontrado, pulando...")
                continue

            if not self.models[model_name].enabled:
                print(f"❌ Modelo {model_name} desabilitado, pulando...")
                continue

            try:
                model_function = self.create_model_function(model_name)
                result = eval_suite.run(model_function)
                individual_results[model_name] = result

                if result and result.summary:
                    score = result.summary.get('overall_score', 0)
                    grade = result.summary.get('overall_grade', 'N/A')
                    print(f"✅ {model_name}: {score:.1f}% ({grade})")
                else:
                    print(f"❌ {model_name}: Falha na avaliação")

            except Exception as e:
                print(f"❌ Erro ao testar {model_name}: {e}")

        # Calcular comparação
        comparison_id = f"comp_{len(model_names)}models_{datetime.now().strftime('%H%M%S')}"
        comparison_result = self._calculate_comparison_metrics(
            comparison_id, suite_name, model_names, individual_results
        )

        return comparison_result

    def _calculate_comparison_metrics(self, comparison_id: str, suite_name: str,
                                    model_names: List[str], individual_results: Dict[str, EvalSuiteResult]) -> ComparisonResult:
        """Calcula métricas de comparação entre modelos."""
        comparison_metrics = {}
        model_scores = {}

        # Extrair scores de cada modelo
        for model_name, result in individual_results.items():
            if result and result.summary:
                model_scores[model_name] = {
                    'overall_score': result.summary.get('overall_score', 0),
                    'overall_grade': result.summary.get('overall_grade', 'F'),
                    'execution_time': result.summary.get('execution_time', 0),
                    'metrics': result.summary.get('metrics', {})
                }

        # Calcular vencedor
        winner = None
        if model_scores:
            winner = max(model_scores.items(), key=lambda x: x[1]['overall_score'])[0]

        # Métricas comparativas
        if len(model_scores) > 1:
            scores = [data['overall_score'] for data in model_scores.values()]
            comparison_metrics = {
                'score_range': max(scores) - min(scores),
                'average_score': sum(scores) / len(scores),
                'best_score': max(scores),
                'worst_score': min(scores),
                'models_above_average': len([s for s in scores if s > sum(scores)/len(scores)])
            }

        # Comparação por métrica
        metric_comparison = {}
        if model_scores:
            # Obter todas as métricas disponíveis
            all_metrics = set()
            for model_data in model_scores.values():
                all_metrics.update(model_data.get('metrics', {}).keys())

            # Comparar cada métrica
            for metric in all_metrics:
                metric_comparison[metric] = {}
                for model_name, model_data in model_scores.items():
                    metric_data = model_data.get('metrics', {}).get(metric, {})
                    if metric_data:
                        metric_comparison[metric][model_name] = metric_data.get('mean', 0)

        comparison_metrics['by_metric'] = metric_comparison

        # Criar resultado
        return ComparisonResult(
            comparison_id=comparison_id,
            timestamp=datetime.now().isoformat(),
            suite_name=suite_name,
            models=list(individual_results.keys()),
            individual_results=individual_results,
            comparison_metrics=comparison_metrics,
            winner=winner,
            summary={
                'total_models': len(individual_results),
                'successful_evaluations': len(model_scores),
                'winner': winner,
                'winner_score': model_scores.get(winner, {}).get('overall_score', 0) if winner else 0,
                'score_gap': comparison_metrics.get('score_range', 0)
            }
        )

    def print_comparison_report(self, result: ComparisonResult):
        """Imprime relatório detalhado de comparação."""
        print(f"\n{'='*60}")
        print(f"🏆 RELATÓRIO DE COMPARAÇÃO DE MODELOS")
        print(f"{'='*60}")

        print(f"📊 Suite: {result.suite_name}")
        print(f"🕒 Timestamp: {result.timestamp}")
        print(f"🤖 Modelos testados: {len(result.models)}")

        if result.winner:
            winner_score = result.summary.get('winner_score', 0)
            print(f"🥇 Vencedor: {result.winner} ({winner_score:.1f}%)")

        print(f"\n📈 SCORES INDIVIDUAIS:")
        print("-" * 40)

        # Ordenar por score
        model_scores = []
        for model_name, eval_result in result.individual_results.items():
            if eval_result and eval_result.summary:
                score = eval_result.summary.get('overall_score', 0)
                grade = eval_result.summary.get('overall_grade', 'F')
                time_taken = eval_result.summary.get('execution_time', 0)
                model_scores.append((model_name, score, grade, time_taken))

        model_scores.sort(key=lambda x: x[1], reverse=True)

        for i, (model, score, grade, time_taken) in enumerate(model_scores, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}️⃣"
            print(f"{medal} {model:<15} {score:>6.1f}% ({grade}) - {time_taken:.2f}s")

        # Comparação por métrica
        if 'by_metric' in result.comparison_metrics:
            print(f"\n📊 COMPARAÇÃO POR MÉTRICA:")
            print("-" * 40)

            for metric, model_scores in result.comparison_metrics['by_metric'].items():
                print(f"\n{metric.upper()}:")
                sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
                for model, score in sorted_models:
                    print(f"  {model:<15} {score:>6.1f}%")

        # Estatísticas gerais
        if 'score_range' in result.comparison_metrics:
            print(f"\n📈 ESTATÍSTICAS:")
            print("-" * 40)
            print(f"Diferença máxima:     {result.comparison_metrics['score_range']:.1f}%")
            print(f"Score médio:          {result.comparison_metrics['average_score']:.1f}%")
            print(f"Melhor performance:   {result.comparison_metrics['best_score']:.1f}%")
            print(f"Pior performance:     {result.comparison_metrics['worst_score']:.1f}%")

        print(f"\n{'='*60}")

    def list_available_models(self):
        """Lista modelos disponíveis para comparação."""
        print(f"\n🤖 MODELOS DISPONÍVEIS:")
        print("-" * 50)

        for name, config in self.models.items():
            status = "✅" if config.enabled else "❌"
            api_status = ""

            if config.api_key_env:
                has_key = bool(os.getenv(config.api_key_env))
                api_status = f" (API: {'✅' if has_key else '❌'})"

            print(f"{status} {name:<12} - {config.description}{api_status}")

    def list_test_suites(self):
        """Lista suites de teste disponíveis."""
        print(f"\n📝 SUITES DE TESTE DISPONÍVEIS:")
        print("-" * 50)

        for name, tests in self.test_suites.items():
            print(f"📋 {name:<12} - {len(tests)} testes")
            for test in tests[:2]:  # Mostrar apenas 2 primeiros
                print(f"   • {test['prompt'][:50]}...")
            if len(tests) > 2:
                print(f"   ... e mais {len(tests)-2} testes")
            print()


def main():
    """Interface principal de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Comparação de Modelos via Sistema de Evals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python utils/model_comparison.py --models gpt4,claude --suite basic
  python utils/model_comparison.py --models local_agent,gpt3.5 --suite reasoning --save-report
  python utils/model_comparison.py --list-models
  python utils/model_comparison.py --list-suites
  python utils/model_comparison.py --interactive
        """
    )

    parser.add_argument("--models", help="Modelos para comparar (separados por vírgula)")
    parser.add_argument("--suite", help="Suite de teste para usar")
    parser.add_argument("--list-models", action="store_true", help="Listar modelos disponíveis")
    parser.add_argument("--list-suites", action="store_true", help="Listar suites de teste disponíveis")
    parser.add_argument("--save-report", action="store_true", help="Salvar relatório em arquivo")
    parser.add_argument("--output-dir", default="comparison_results", help="Diretório para salvar resultados")
    parser.add_argument("--interactive", action="store_true", help="Modo interativo")

    args = parser.parse_args()

    if not EVALS_AVAILABLE:
        print("❌ Sistema de evals não está disponível")
        print("Verifique se todas as dependências estão instaladas")
        return 1

    # Criar framework
    framework = ModelComparisonFramework()

    # Listar modelos
    if args.list_models:
        framework.list_available_models()
        return 0

    # Listar suites
    if args.list_suites:
        framework.list_test_suites()
        return 0

    # Modo interativo
    if args.interactive:
        return interactive_mode(framework)

    # Validar argumentos para comparação
    if not args.models or not args.suite:
        parser.print_help()
        print("\n❌ Especifique --models e --suite para executar comparação")
        return 1

    # Executar comparação
    try:
        model_names = [m.strip() for m in args.models.split(',')]
        result = framework.compare_models(model_names, args.suite)

        # Imprimir relatório
        framework.print_comparison_report(result)

        # Salvar resultado se solicitado
        if args.save_report:
            os.makedirs(args.output_dir, exist_ok=True)
            file_path = os.path.join(args.output_dir, f"comparison_{result.comparison_id}.json")
            result.save_to_file(file_path)

        return 0

    except Exception as e:
        print(f"❌ Erro durante comparação: {e}")
        return 1


def interactive_mode(framework: ModelComparisonFramework) -> int:
    """Modo interativo para seleção de modelos e suites."""
    print("🎯 MODO INTERATIVO - COMPARAÇÃO DE MODELOS")
    print("=" * 50)

    # Mostrar opções
    framework.list_available_models()
    framework.list_test_suites()

    # Selecionar modelos
    print("\n🤖 Selecione modelos para comparar:")
    available_models = [name for name, config in framework.models.items() if config.enabled]

    for i, model in enumerate(available_models, 1):
        print(f"  {i}. {model}")

    try:
        selection = input("\nDigite números separados por vírgula (ex: 1,3): ").strip()
        model_indices = [int(x.strip()) - 1 for x in selection.split(',')]
        selected_models = [available_models[i] for i in model_indices if 0 <= i < len(available_models)]

        if len(selected_models) < 2:
            print("❌ Selecione pelo menos 2 modelos para comparação")
            return 1

    except (ValueError, IndexError):
        print("❌ Seleção inválida")
        return 1

    # Selecionar suite
    print(f"\n📝 Selecione suite de teste:")
    suites = list(framework.test_suites.keys())

    for i, suite in enumerate(suites, 1):
        print(f"  {i}. {suite} ({len(framework.test_suites[suite])} testes)")

    try:
        suite_index = int(input("\nDigite o número da suite: ").strip()) - 1
        selected_suite = suites[suite_index]
    except (ValueError, IndexError):
        print("❌ Seleção inválida")
        return 1

    # Confirmar e executar
    print(f"\n✅ Configuração:")
    print(f"   Modelos: {', '.join(selected_models)}")
    print(f"   Suite: {selected_suite}")

    confirm = input("\nProsseguir com a comparação? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Comparação cancelada")
        return 0

    # Executar comparação
    try:
        result = framework.compare_models(selected_models, selected_suite)
        framework.print_comparison_report(result)

        save = input("\nSalvar relatório? (y/n): ").strip().lower()
        if save == 'y':
            result.save_to_file()

        return 0

    except Exception as e:
        print(f"❌ Erro durante comparação: {e}")
        return 1


if __name__ == "__main__":
    exit(main())