#!/usr/bin/env python3
"""
Exemplos de comparação de modelos específicos.
Demonstra como comparar Llama vs Gemini e modelos Ollama.
"""

import os
import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_comparison import ModelComparisonFramework


def example_llama_vs_gemini():
    """Exemplo: Comparar Llama 3 (local) vs Gemini 1.5 (cloud)."""
    print("\n🔥 EXEMPLO 1: LLAMA 3 vs GEMINI 1.5")
    print("=" * 50)

    framework = ModelComparisonFramework()

    # Verificar se modelos estão disponíveis
    print("📋 Verificando disponibilidade dos modelos...")

    # Verificar Gemini API Key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key or gemini_key == "AIzaSyBTTv0AW2t3Zwth742xwDo2Yyw3f7_QytQ":
        print("⚠️  GEMINI_API_KEY não configurada ou usando valor de exemplo")
        print("   Configure uma API key válida no .env para usar Gemini")
        return False

    # Verificar Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            if "llama3:latest" not in model_names:
                print("⚠️  Modelo llama3:latest não encontrado no Ollama")
                print("   Execute: ollama pull llama3")
                return False
            print("✅ Ollama conectado e llama3:latest disponível")
        else:
            print("❌ Ollama não está respondendo em localhost:11434")
            return False
    except Exception as e:
        print(f"❌ Erro ao conectar com Ollama: {e}")
        return False

    # Executar comparação
    try:
        print("\n🚀 Iniciando comparação...")
        result = framework.compare_models(
            ["llama3_ollama", "gemini_15_flash"],
            "basic"
        )

        # Mostrar relatório
        framework.print_comparison_report(result)

        # Salvar resultado
        result.save_to_file("comparison_llama_vs_gemini.json")

        return True

    except Exception as e:
        print(f"❌ Erro durante comparação: {e}")
        return False


def example_ollama_models():
    """Exemplo: Comparar modelos Ollama (Llama 3 vs GPT-OSS)."""
    print("\n🔥 EXEMPLO 2: MODELOS OLLAMA (LLAMA 3 vs GPT-OSS)")
    print("=" * 60)

    framework = ModelComparisonFramework()

    # Verificar se modelos estão disponíveis no Ollama
    print("📋 Verificando modelos no Ollama...")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]

            print(f"🔍 Modelos encontrados no Ollama:")
            for name in model_names:
                print(f"   • {name}")

            # Verificar se os modelos necessários estão disponíveis
            required_models = ["llama3:latest", "gpt-oss:latest"]
            available = []

            for model in required_models:
                if model in model_names:
                    print(f"✅ {model} - Disponível")
                    available.append(model)
                else:
                    print(f"❌ {model} - NÃO encontrado")
                    print(f"   Execute: ollama pull {model.split(':')[0]}")

            if len(available) < 2:
                print("\n⚠️  Pelo menos 2 modelos são necessários para comparação")
                return False

        else:
            print("❌ Ollama não está respondendo")
            return False

    except Exception as e:
        print(f"❌ Erro ao conectar com Ollama: {e}")
        return False

    # Executar comparação
    try:
        print("\n🚀 Iniciando comparação entre modelos Ollama...")
        result = framework.compare_models(
            ["llama3_ollama", "gpt_oss_ollama"],
            "reasoning"
        )

        # Mostrar relatório
        framework.print_comparison_report(result)

        # Salvar resultado
        result.save_to_file("comparison_ollama_models.json")

        return True

    except Exception as e:
        print(f"❌ Erro durante comparação: {e}")
        return False


def check_requirements():
    """Verifica dependências necessárias."""
    print("🔍 VERIFICANDO DEPENDÊNCIAS")
    print("=" * 30)

    # Verificar Python packages
    required_packages = [
        ("requests", "Para comunicação com Ollama"),
        ("google-generativeai", "Para uso do Gemini (opcional)"),
    ]

    missing_packages = []

    for package, description in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n📦 Instale os packages necessários:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    # Verificar serviços
    print(f"\n🔍 VERIFICANDO SERVIÇOS")
    print("-" * 25)

    # Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=3)
        if response.status_code == 200:
            version_info = response.json()
            print(f"✅ Ollama v{version_info.get('version', 'unknown')} - Rodando")
        else:
            print("❌ Ollama - Não está respondendo")
            return False
    except Exception:
        print("❌ Ollama - Não está rodando ou não está instalado")
        print("   Instale: https://ollama.ai/")
        return False

    # API Keys
    print(f"\n🔑 VERIFICANDO API KEYS")
    print("-" * 22)

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and gemini_key != "AIzaSyBTTv0AW2t3Zwth742xwDo2Yyw3f7_QytQ":
        print("✅ GEMINI_API_KEY - Configurada")
    else:
        print("⚠️  GEMINI_API_KEY - Não configurada ou usando valor de exemplo")

    return True


def interactive_menu():
    """Menu interativo para seleção de exemplos."""
    while True:
        print("\n🎯 MENU DE COMPARAÇÃO DE MODELOS")
        print("=" * 35)
        print("1. 🔥 Llama 3 vs Gemini 1.5")
        print("2. 🔥 Modelos Ollama (Llama vs GPT-OSS)")
        print("3. 🔍 Verificar dependências")
        print("4. 📋 Listar modelos disponíveis")
        print("5. ❌ Sair")

        try:
            choice = input("\n➤ Escolha uma opção (1-5): ").strip()

            if choice == "1":
                example_llama_vs_gemini()
            elif choice == "2":
                example_ollama_models()
            elif choice == "3":
                check_requirements()
            elif choice == "4":
                framework = ModelComparisonFramework()
                framework.list_available_models()
            elif choice == "5":
                print("👋 Saindo...")
                break
            else:
                print("❌ Opção inválida")

            input("\n⏸️  Pressione Enter para continuar...")

        except KeyboardInterrupt:
            print("\n👋 Saindo...")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")


def main():
    """Função principal."""
    print("🚀 SISTEMA DE COMPARAÇÃO DE MODELOS")
    print("Agent Core - Eval System")
    print("=" * 40)

    # Verificar dependências básicas
    if not check_requirements():
        print("\n❌ Dependências não satisfeitas. Corrija os problemas acima.")
        return 1

    # Menu interativo
    interactive_menu()

    return 0


if __name__ == "__main__":
    exit(main())