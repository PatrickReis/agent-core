"""
Módulo de utilitários.

Contém utilitários auxiliares para o projeto, incluindo conversão automática
de especificações OpenAPI/Swagger em ferramentas LangGraph funcionais.
Facilita a integração com APIs externas.

Utilitários principais:
- OpenAPIToLangGraphTools: Conversor OpenAPI → LangGraph Tools
- Geração automática de código Python com decorators @tool
- Suporte a múltiplos métodos HTTP (GET, POST, PUT, DELETE, PATCH)
- Validação de parâmetros e tratamento de erros
"""