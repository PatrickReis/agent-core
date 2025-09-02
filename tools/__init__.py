"""
Módulo de ferramentas LangChain.

Contém as ferramentas disponíveis para o agente, incluindo busca vetorial
na base de conhecimento ChromaDB, informações meteorológicas e utilitários.
Suporte a geração automática de tools a partir de especificações OpenAPI.

Ferramentas disponíveis:
- search_knowledge_base: Busca na base vetorial ChromaDB
- get_weather: Informações meteorológicas via Open-Meteo API
- simple_calculator: Calculadora básica
- echo_tool: Ferramenta de teste/debug
"""