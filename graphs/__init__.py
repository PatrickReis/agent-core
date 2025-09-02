"""
Módulo de grafos LangGraph.

Este módulo contém a implementação do grafo inteligente que orquestra
o comportamento do agente. O grafo decide automaticamente quando usar
ferramentas específicas (busca vetorial, clima) versus resposta direta,
baseado na análise da pergunta do usuário.

Componentes principais:
- AgentState: Estado do agente com histórico de mensagens
- Nós de decisão: should_continue, call_model, call_tool
- Orquestração inteligente: análise de palavras-chave para determinar fluxo
"""