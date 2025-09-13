#!/usr/bin/env python3
"""
Script de teste para as funcionalidades do MCP Server Enhanced.
Demonstra como usar todas as funcionalidades disponíveis.
"""

import asyncio
import json
from typing import Dict, List, Any

# Simular cliente MCP para testes
class MCPTestClient:
    """Cliente de teste para simular chamadas MCP."""

    def __init__(self, server_url: str = "local"):
        self.server_url = server_url
        self.session_id = "test_session"

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Simula chamada de tool MCP."""
        print(f"📡 Chamando tool: {tool_name}")
        print(f"📋 Argumentos: {json.dumps(arguments, indent=2, ensure_ascii=False)}")

        # Aqui você conectaria ao servidor MCP real
        # Por enquanto, retornamos dados simulados
        return {"result": "simulado", "tool": tool_name, "args": arguments}

    async def get_resource(self, resource_uri: str) -> Dict[str, Any]:
        """Simula obtenção de resource MCP."""
        print(f"📦 Obtendo resource: {resource_uri}")
        return {"resource": resource_uri, "data": "simulado"}


async def test_evaluation_system():
    """Testa o sistema de avaliação de prompts."""
    print("\n🧪 === TESTANDO SISTEMA DE AVALIAÇÃO ===")

    client = MCPTestClient()

    # Teste 1: Avaliação rápida
    test_prompts = [
        "Qual é a temperatura em São Paulo?",
        "Explique o que é Python",
        "Como está o clima hoje?",
        "O que significa IA?"
    ]

    print("\n1️⃣ Teste de Avaliação Rápida:")
    result = await client.call_tool("run_agent_evaluation", {
        "test_prompts": test_prompts,
        "suite_name": "quick_test"
    })
    print(f"✅ Resultado: {json.dumps(result, indent=2, ensure_ascii=False)}")


async def test_advanced_reasoning():
    """Testa o sistema de reasoning avançado."""
    print("\n🧠 === TESTANDO REASONING AVANÇADO ===")

    client = MCPTestClient()

    # Teste com diferentes níveis de complexidade
    test_questions = [
        {
            "question": "Por que o céu é azul?",
            "complexity": "simple"
        },
        {
            "question": "Como funciona o machine learning e qual sua aplicação em sistemas de recomendação?",
            "complexity": "complex"
        },
        {
            "question": "Analise os prós e contras da implementação de microserviços vs arquitetura monolítica",
            "complexity": "advanced"
        }
    ]

    for i, test in enumerate(test_questions, 1):
        print(f"\n{i}️⃣ Teste de Reasoning - {test['complexity'].upper()}:")
        print(f"Pergunta: {test['question']}")

        result = await client.call_tool("enhanced_reasoning", {
            "question": test["question"],
            "complexity_override": test["complexity"]
        })
        print(f"✅ Resultado: {json.dumps(result, indent=2, ensure_ascii=False)[:200]}...")


async def test_feature_management():
    """Testa o gerenciamento de features."""
    print("\n⚙️ === TESTANDO GERENCIAMENTO DE FEATURES ===")

    client = MCPTestClient()

    # Listar features habilitadas
    print("\n1️⃣ Listando features habilitadas:")
    result = await client.call_tool("manage_features", {"action": "list"})
    print(f"✅ Features: {json.dumps(result, indent=2, ensure_ascii=False)}")

    # Ver status completo
    print("\n2️⃣ Status completo das features:")
    result = await client.call_tool("manage_features", {"action": "status"})
    print(f"✅ Status: {json.dumps(result, indent=2, ensure_ascii=False)}")

    # Teste de habilitação/desabilitação (apenas simulado)
    print("\n3️⃣ Simulando toggle de feature:")
    print("❗ Em produção, use com cuidado - pode afetar funcionalidades")


async def test_resources():
    """Testa os resources do sistema."""
    print("\n📊 === TESTANDO RESOURCES ===")

    client = MCPTestClient()

    # Configuração do sistema
    print("\n1️⃣ Configuração Enhanced:")
    config = await client.get_resource("config://enhanced")
    print(f"⚙️ Config: {json.dumps(config, indent=2, ensure_ascii=False)}")

    # Métricas do sistema
    print("\n2️⃣ Métricas do Sistema:")
    metrics = await client.get_resource("metrics://system")
    print(f"📈 Métricas: {json.dumps(metrics, indent=2, ensure_ascii=False)}")

    # Health check
    print("\n3️⃣ Health Check:")
    health = await client.get_resource("health://check")
    print(f"🏥 Saúde: {json.dumps(health, indent=2, ensure_ascii=False)}")


async def test_reasoning_traces():
    """Testa obtenção de traces de reasoning."""
    print("\n🔍 === TESTANDO REASONING TRACES ===")

    client = MCPTestClient()

    # Primeiro, executar reasoning para gerar trace
    print("1️⃣ Gerando trace de reasoning:")
    reasoning_result = await client.call_tool("enhanced_reasoning", {
        "question": "Qual a diferença entre Python e JavaScript?"
    })

    if "trace_id" in reasoning_result.get("result", {}):
        trace_id = reasoning_result["result"]["trace_id"]

        # Obter trace detalhado
        print(f"\n2️⃣ Obtendo trace detalhado: {trace_id}")
        trace_result = await client.call_tool("get_reasoning_trace", {
            "trace_id": trace_id
        })
        print(f"🔍 Trace: {json.dumps(trace_result, indent=2, ensure_ascii=False)}")
    else:
        print("❌ Não foi possível obter trace_id")


def create_eval_examples():
    """Cria exemplos práticos de uso do sistema de evals."""
    print("\n📝 === EXEMPLOS DE USO DOS EVALS ===")

    examples = {
        "eval_basico": {
            "descricao": "Avaliação básica com poucos prompts",
            "prompts": [
                "O que é Python?",
                "Como está o tempo hoje?",
                "Explique machine learning"
            ],
            "uso": "await client.call_tool('run_agent_evaluation', {'test_prompts': prompts, 'suite_name': 'basico'})"
        },

        "eval_completo": {
            "descricao": "Suite completa com casos de teste estruturados",
            "exemplo_codigo": """
# Exemplo de uso programático do framework de evals
from utils.prompt_evals import EvalFramework, create_standard_eval_suite

# Criar framework
framework = EvalFramework()

# Definir casos de teste
test_cases = [
    {
        "id": "python_concept",
        "prompt": "O que é Python?",
        "expected": "Python é uma linguagem de programação",
        "tags": ["conceito", "programação"]
    },
    {
        "id": "weather_query",
        "prompt": "Qual a temperatura em SP?",
        "tags": ["clima", "tool_usage"]
    }
]

# Criar suite
suite = create_standard_eval_suite("minha_suite", test_cases)

# Executar com função do agente
def meu_agente(prompt):
    # Sua lógica aqui
    return "resposta do agente"

result = suite.run(meu_agente)
print(f"Score: {result.summary['overall_score']:.1f}%")
"""
        },

        "metricas_disponiveis": {
            "descricao": "Métricas avaliadas automaticamente",
            "metricas": {
                "relevance": "Relevância da resposta ao prompt (0-100%)",
                "accuracy": "Precisão comparada com resposta esperada (0-100%)",
                "latency": "Tempo de resposta (baseado em target de 2s)",
                "safety": "Segurança - detecta conteúdo perigoso (0-100%)",
                "tool_usage": "Uso apropriado de ferramentas (0-100%)"
            }
        }
    }

    for nome, exemplo in examples.items():
        print(f"\n📋 {nome.upper()}:")
        print(f"   {exemplo['descricao']}")

        if "prompts" in exemplo:
            print(f"   Prompts: {exemplo['prompts']}")
            print(f"   Uso: {exemplo['uso']}")

        if "exemplo_codigo" in exemplo:
            print(f"   Código de exemplo:")
            print(exemplo["exemplo_codigo"])

        if "metricas" in exemplo:
            print("   Métricas avaliadas:")
            for metrica, desc in exemplo["metricas"].items():
                print(f"   • {metrica}: {desc}")


def show_available_features():
    """Mostra todas as funcionalidades disponíveis."""
    print("\n🚀 === FUNCIONALIDADES DISPONÍVEIS ===")

    features = {
        "Tools MCP": {
            "run_agent_evaluation": "Executa avaliação de prompts com múltiplas métricas",
            "enhanced_reasoning": "Reasoning multi-step com análise complexa",
            "manage_features": "Gerencia feature toggles do sistema",
            "get_reasoning_trace": "Obtém traces detalhados de reasoning"
        },

        "Resources MCP": {
            "config://enhanced": "Configuração completa do servidor",
            "metrics://system": "Métricas básicas do sistema",
            "metrics://detailed": "Métricas detalhadas (admin)",
            "health://check": "Health check completo"
        },

        "Prompts MCP": {
            "enhanced-agent-query": "Template para consultas ao agente",
            "evaluation-prompt": "Template para cenários de teste"
        },

        "Features Habilitadas": {
            "prompt_evaluation": "Sistema de avaliação de prompts",
            "advanced_reasoning": "Reasoning avançado multi-step",
            "enhanced_logging": "Logging estruturado com métricas",
            "performance_monitoring": "Monitoramento de performance"
        }
    }

    for categoria, items in features.items():
        print(f"\n📦 {categoria}:")
        for nome, desc in items.items():
            print(f"   🔹 {nome}: {desc}")


async def main():
    """Executa todos os testes das funcionalidades."""
    print("🎯 TESTANDO MCP SERVER ENHANCED")
    print("=" * 50)

    # Mostrar funcionalidades disponíveis
    show_available_features()

    # Criar exemplos de uso
    create_eval_examples()

    # Executar testes das funcionalidades
    await test_evaluation_system()
    await test_advanced_reasoning()
    await test_feature_management()
    await test_resources()
    await test_reasoning_traces()

    print("\n" + "=" * 50)
    print("✅ TESTES CONCLUÍDOS!")
    print("\n💡 PRÓXIMOS PASSOS:")
    print("1. Conecte um cliente MCP real ao seu servidor")
    print("2. Execute as tools através do cliente MCP")
    print("3. Monitore logs do servidor para ver execução real")
    print("4. Use os resources para verificar status e métricas")
    print("5. Experimente diferentes cenários de avaliação")


if __name__ == "__main__":
    asyncio.run(main())