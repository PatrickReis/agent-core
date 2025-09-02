"""
Módulo de logging profissional.

Sistema de logging estruturado com múltiplos níveis e saídas coloridas
para console e arquivo. Otimizado para compatibilidade MCP e debugging
de aplicações com múltiplos componentes.

Características:
- Loggers especializados por componente (mcp, llm, agent, tools, bridge)
- Formatação profissional com timestamps e contexto
- Saída para console (stderr) e arquivo persistente
- Métodos especializados: success, progress, tool_execution, etc.
"""