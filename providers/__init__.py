"""
Módulo de provedores LLM.

Sistema modular para gerenciar diferentes provedores de modelos de linguagem
(Ollama, OpenAI, Google Gemini). Permite alternar entre provedores via
configuração de ambiente sem modificar código.

Funcionalidades:
- Factory pattern para criação de provedores
- Configuração automática via variáveis de ambiente
- Suporte a múltiplos provedores LLM e embeddings
- Detecção automática de dependências
"""