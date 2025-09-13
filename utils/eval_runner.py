#!/usr/bin/env python3
"""
Eval Runner - Interface CLI para execução de avaliações individuais.
Complementa o model_comparison.py focando em execução de evals únicos.

Usage:
    python utils/eval_runner.py --agent local --suite basic
    python utils/eval_runner.py --custom-prompts prompts.txt --save
    python utils/eval_runner.py --quick "Pergunta 1,Pergunta 2,Pergunta 3"
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Adicionar o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.prompt_evals import EvalFramework, create_standard_eval_suite, run_quick_eval
    from utils.enhanced_logging import get_enhanced_logger
    from utils.feature_toggles import is_feature_enabled
    EVALS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Sistema de evals não disponível: {e}")
    EVALS_AVAILABLE = False

logger = get_enhanced_logger("eval_runner") if EVALS_AVAILABLE else None


class EvalRunner:
    """Runner para execução de avaliações individuais."""

    def __init__(self):
        self.eval_framework = EvalFramework() if EVALS_AVAILABLE else None
        self.agent_configs = self._load_agent_configs()

    def _load_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """Carrega configurações de agentes disponíveis."""
        return {
            "local": {
                "name": "Agent Core Local",
                "description": "Agente local do Agent Core via LangGraph",
                "type": "local"
            },
            "mock": {
                "name": "Mock Agent",
                "description": "Agente simulado para testes",
                "type": "mock"
            }
        }

    def create_agent_function(self, agent_type: str):
        """Cria função de agente baseada no tipo."""
        if agent_type == "local":
            return self._create_local_agent()
        elif agent_type == "mock":
            return self._create_mock_agent()
        else:
            raise ValueError(f"Tipo de agente não suportado: {agent_type}")

    def _create_local_agent(self):
        """Cria função para agente local."""
        def local_agent_function(prompt: str) -> str:
            try:
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

        return local_agent_function

    def _create_mock_agent(self):
        """Cria função para agente mock."""
        def mock_agent_function(prompt: str) -> str:
            # Simular processamento
            time.sleep(0.1)

            # Respostas baseadas no prompt
            prompt_lower = prompt.lower()

            if any(word in prompt_lower for word in ["python", "programação", "código"]):
                return "Python é uma linguagem de programação interpretada, de alto nível e de uso geral, conhecida por sua sintaxe clara e legível."

            elif any(word in prompt_lower for word in ["clima", "temperatura", "tempo"]):
                return "O clima hoje está ensolarado com temperatura agradável de 24°C, ideal para atividades ao ar livre."

            elif any(word in prompt_lower for word in ["machine learning", "ml", "inteligência artificial", "ia"]):
                return "Machine Learning é um subcampo da inteligência artificial que permite que sistemas aprendam e melhorem automaticamente a partir de dados sem serem explicitamente programados."

            elif any(word in prompt_lower for word in ["brasil", "capital", "brasília"]):
                return "A capital do Brasil é Brasília, localizada na região Centro-Oeste do país e inaugurada em 1960."

            elif any(word in prompt_lower for word in ["círculo", "área", "matemática"]):
                return "A área de um círculo é calculada pela fórmula A = π × r², onde r é o raio do círculo."

            else:
                return f"Esta é uma resposta informativa e contextualizada para sua pergunta sobre: {prompt[:50]}..."

        return mock_agent_function

    def run_suite_evaluation(self, agent_type: str, suite_name: str, custom_tests: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Executa avaliação usando uma suite específica."""
        if not EVALS_AVAILABLE:
            raise ValueError("Sistema de evals não disponível")

        print(f"\n🧪 Executando avaliação com suite: {suite_name}")
        print(f"🤖 Agente: {self.agent_configs[agent_type]['name']}")

        # Definir casos de teste
        if custom_tests:
            test_cases = custom_tests
        else:
            test_cases = self._get_default_suite(suite_name)

        if not test_cases:
            raise ValueError(f"Suite {suite_name} não encontrada ou vazia")

        print(f"📝 Total de testes: {len(test_cases)}")

        # Criar suite de avaliação
        eval_suite = create_standard_eval_suite(f"{agent_type}_{suite_name}", test_cases)

        # Criar função do agente
        agent_function = self.create_agent_function(agent_type)

        # Executar avaliação
        start_time = time.time()
        result = eval_suite.run(agent_function)
        execution_time = time.time() - start_time

        if result:
            print(f"\n✅ Avaliação concluída em {execution_time:.2f}s")
            print(f"📊 Score geral: {result.summary['overall_score']:.1f}% (Nota: {result.summary['overall_grade']})")

            # Mostrar métricas detalhadas
            print(f"\n📈 Métricas por evaluator:")
            for metric, data in result.summary['metrics'].items():
                pass_rate = data['pass_rate']
                mean_score = data['mean']
                status_icon = "✅" if pass_rate >= 70 else "⚠️" if pass_rate >= 50 else "❌"
                print(f"  {status_icon} {metric:12} {mean_score:6.1f}% (aprovação: {pass_rate:5.1f}%)")

            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "summary": {
                    "agent": agent_type,
                    "suite": suite_name,
                    "total_tests": len(test_cases),
                    "overall_score": result.summary['overall_score'],
                    "overall_grade": result.summary['overall_grade'],
                    "metrics": result.summary['metrics']
                }
            }
        else:
            return {
                "success": False,
                "error": "Avaliação falhou - feature possivelmente desabilitada",
                "execution_time": execution_time
            }

    def run_quick_evaluation(self, agent_type: str, prompts: List[str]) -> Dict[str, Any]:
        """Executa avaliação rápida com lista de prompts."""
        if not EVALS_AVAILABLE:
            raise ValueError("Sistema de evals não disponível")

        print(f"\n⚡ Avaliação rápida")
        print(f"🤖 Agente: {self.agent_configs[agent_type]['name']}")
        print(f"📝 Prompts: {len(prompts)}")

        # Criar função do agente
        agent_function = self.create_agent_function(agent_type)

        # Executar avaliação rápida
        start_time = time.time()
        metrics = run_quick_eval(prompts, agent_function)
        execution_time = time.time() - start_time

        if metrics:
            overall_score = sum(metrics.values()) / len(metrics)
            print(f"\n✅ Avaliação rápida concluída em {execution_time:.2f}s")
            print(f"📊 Score médio: {overall_score:.1f}%")

            # Mostrar métricas
            print(f"\n📈 Métricas:")
            for metric, score in metrics.items():
                status_icon = "✅" if score >= 70 else "⚠️" if score >= 50 else "❌"
                print(f"  {status_icon} {metric:12} {score:6.1f}%")

            return {
                "success": True,
                "metrics": metrics,
                "overall_score": overall_score,
                "execution_time": execution_time,
                "prompts_count": len(prompts)
            }
        else:
            return {
                "success": False,
                "error": "Avaliação rápida falhou",
                "execution_time": execution_time
            }

    def _get_default_suite(self, suite_name: str) -> List[Dict[str, Any]]:
        """Retorna casos de teste para suites padrão."""
        suites = {
            "basic": [
                {"prompt": "O que é Python?", "expected": "Python é uma linguagem de programação"},
                {"prompt": "Qual a capital do Brasil?", "expected": "Brasília"},
                {"prompt": "Como calcular a área de um círculo?"},
                {"prompt": "Explique o conceito de IA em uma frase"}
            ],

            "reasoning": [
                {"prompt": "Por que o céu é azul? Explique de forma simples."},
                {"prompt": "Se chove quando há nuvens escuras, e há nuvens escuras agora, vai chover?"},
                {"prompt": "Compare os prós e contras de energia solar"},
                {"prompt": "Qual a relação entre causa e efeito na programação?"}
            ],

            "knowledge": [
                {"prompt": "Explique machine learning em duas frases"},
                {"prompt": "Qual a diferença entre HTTP e HTTPS?"},
                {"prompt": "Como funciona a internet?"},
                {"prompt": "O que são APIs e para que servem?"}
            ],

            "performance": [
                {"prompt": "Resposta rápida: 2+2"},
                {"prompt": "Uma palavra: capital da França"},
                {"prompt": "Sim ou não: Python é compilado?"},
                {"prompt": "Nome de uma cor primária"}
            ],

            "comprehensive": [
                # Mistura de todos os tipos
                {"prompt": "O que é Python?", "expected": "Python é uma linguagem de programação"},
                {"prompt": "Por que o céu é azul?"},
                {"prompt": "Explique machine learning brevemente"},
                {"prompt": "Capital do Brasil?", "expected": "Brasília"},
                {"prompt": "Como calcular área de círculo?"},
                {"prompt": "Diferença entre HTTP e HTTPS?"},
                {"prompt": "2+2 = ?", "expected": "4"},
                {"prompt": "Python é compilado?", "expected": "Não"}
            ]
        }

        return suites.get(suite_name, [])

    def load_prompts_from_file(self, file_path: str) -> List[str]:
        """Carrega prompts de arquivo texto."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        prompts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Ignorar comentários
                    prompts.append(line)

        return prompts

    def save_results(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Salva resultados em arquivo JSON."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"eval_results_{timestamp}.json"

        # Preparar dados para serialização
        serializable_results = {}
        for key, value in results.items():
            if hasattr(value, 'to_dict'):
                serializable_results[key] = value.to_dict()
            elif hasattr(value, '__dict__'):
                serializable_results[key] = value.__dict__
            else:
                serializable_results[key] = value

        # Salvar arquivo
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)

        print(f"💾 Resultados salvos em: {output_file}")
        return output_file

    def list_available_suites(self):
        """Lista suites de teste disponíveis."""
        print(f"\n📝 SUITES DE TESTE DISPONÍVEIS:")
        print("-" * 50)

        suites_info = {
            "basic": "Testes básicos de conhecimento geral",
            "reasoning": "Testes de raciocínio e lógica",
            "knowledge": "Testes de conhecimento técnico",
            "performance": "Testes de velocidade de resposta",
            "comprehensive": "Suite completa com todos os tipos"
        }

        for name, description in suites_info.items():
            test_count = len(self._get_default_suite(name))
            print(f"📋 {name:<14} - {description} ({test_count} testes)")

    def list_available_agents(self):
        """Lista agentes disponíveis."""
        print(f"\n🤖 AGENTES DISPONÍVEIS:")
        print("-" * 50)

        for name, config in self.agent_configs.items():
            print(f"🤖 {name:<10} - {config['description']}")


def main():
    """Interface principal de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Eval Runner - Execução de avaliações individuais",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python utils/eval_runner.py --agent local --suite basic
  python utils/eval_runner.py --agent mock --suite reasoning --save
  python utils/eval_runner.py --agent local --quick "O que é Python?,Capital do Brasil?"
  python utils/eval_runner.py --agent mock --custom-prompts my_prompts.txt
  python utils/eval_runner.py --list-suites
  python utils/eval_runner.py --list-agents
        """
    )

    parser.add_argument("--agent", choices=["local", "mock"], default="mock",
                       help="Tipo de agente para avaliar")
    parser.add_argument("--suite", help="Suite de teste padrão para usar")
    parser.add_argument("--quick", help="Prompts separados por vírgula para avaliação rápida")
    parser.add_argument("--custom-prompts", help="Arquivo com prompts personalizados")
    parser.add_argument("--save", action="store_true", help="Salvar resultados em arquivo")
    parser.add_argument("--output", help="Arquivo de saída para resultados")
    parser.add_argument("--list-suites", action="store_true", help="Listar suites disponíveis")
    parser.add_argument("--list-agents", action="store_true", help="Listar agentes disponíveis")

    args = parser.parse_args()

    if not EVALS_AVAILABLE:
        print("❌ Sistema de evals não está disponível")
        return 1

    # Criar runner
    runner = EvalRunner()

    # Listar suites
    if args.list_suites:
        runner.list_available_suites()
        return 0

    # Listar agentes
    if args.list_agents:
        runner.list_available_agents()
        return 0

    # Validar que pelo menos uma opção de teste foi especificada
    if not any([args.suite, args.quick, args.custom_prompts]):
        parser.print_help()
        print("\n❌ Especifique --suite, --quick ou --custom-prompts")
        return 1

    try:
        results = None

        # Avaliação rápida
        if args.quick:
            prompts = [p.strip() for p in args.quick.split(',')]
            results = runner.run_quick_evaluation(args.agent, prompts)

        # Prompts de arquivo
        elif args.custom_prompts:
            prompts = runner.load_prompts_from_file(args.custom_prompts)
            print(f"📁 Carregados {len(prompts)} prompts de {args.custom_prompts}")
            results = runner.run_quick_evaluation(args.agent, prompts)

        # Suite padrão
        elif args.suite:
            results = runner.run_suite_evaluation(args.agent, args.suite)

        # Salvar resultados se solicitado
        if results and args.save:
            runner.save_results(results, args.output)

        return 0 if results and results.get('success', False) else 1

    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        return 1


if __name__ == "__main__":
    exit(main())