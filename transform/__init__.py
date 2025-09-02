"""
Módulo de transformação LangChain ↔ MCP.

Bridge que converte ferramentas LangChain para o protocolo MCP (Model Context Protocol),
permitindo expor agentes LangGraph completos como ferramentas MCP via FastMCP.
Inclui wrappers assíncronos e cache para otimização.

Funcionalidades:
- Conversão automática de LangChain Tools → MCP Tools
- Exposição de agentes LangGraph como tools MCP
- Suporte a tools estruturadas e simples
- Cache de agentes para melhor performance
"""