"""
Sistema de Feature Toggles para Agent Core.
Controla funcionalidades de forma granular e dinâmica.

Este sistema permite:
- Habilitar/desabilitar features via configuração
- Rollout percentage para releases graduais
- User whitelist para testes específicos
- Dependencies entre features
- Configuração por ambiente (dev/staging/prod)
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class FeatureStatus(Enum):
    """Status possíveis para uma feature toggle."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    PERCENTAGE = "percentage"
    WHITELIST = "whitelist"


class FeatureConfig:
    """Configuração de uma feature toggle."""

    def __init__(
        self,
        name: str,
        enabled: bool = False,
        description: str = "",
        environments: List[str] = None,
        rollout_percentage: int = 0,
        user_whitelist: List[str] = None,
        dependencies: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        self.name = name
        self.enabled = enabled
        self.description = description
        self.environments = environments or ["dev", "staging", "prod"]
        self.rollout_percentage = max(0, min(100, rollout_percentage))
        self.user_whitelist = user_whitelist or []
        self.dependencies = dependencies or []
        self.metadata = metadata or {}


class FeatureToggleManager:
    """
    Gerenciador principal do sistema de Feature Toggles.

    Exemplos de uso:

    # Configuração básica
    feature_manager = FeatureToggleManager()

    # Verificar se feature está habilitada
    if feature_manager.is_enabled("oauth2_authentication"):
        # Código protegido por OAuth2
        pass

    # Usar decorators
    @feature_manager.require_feature("prompt_evaluation")
    def run_evaluation():
        # Código que só executa se feature estiver habilitada
        pass
    """

    def __init__(self, config_file: Optional[str] = None, environment: Optional[str] = None):
        self.environment = environment or os.getenv("ENVIRONMENT", "dev")
        self.config_file = config_file or "feature_toggles.json"
        self.features: Dict[str, FeatureConfig] = {}
        self._load_configuration()

        logger.info(f"FeatureToggleManager inicializado para ambiente: {self.environment}")

    def _load_configuration(self):
        """Carrega configuração de features de arquivo JSON e variáveis de ambiente."""
        # Carregar configuração padrão
        self._load_default_features()

        # Carregar do arquivo JSON se existir
        if Path(self.config_file).exists():
            self._load_from_file()

        # Sobrescrever com variáveis de ambiente
        self._load_from_env()

        # Validar dependencies
        self._validate_dependencies()

    def _load_default_features(self):
        """Carrega features padrão do sistema."""
        default_features = [
            FeatureConfig(
                name="prompt_evaluation",
                enabled=self.environment in ["dev", "staging"],
                description="Sistema de avaliação de prompts e qualidade",
                environments=["dev", "staging"],
                metadata={"category": "evaluation", "impact": "low"}
            ),
            FeatureConfig(
                name="advanced_reasoning",
                enabled=True,
                description="Sistema de reasoning multi-step com reflexão",
                environments=["dev", "staging", "prod"],
                metadata={"category": "core", "impact": "high"}
            ),
            FeatureConfig(
                name="oauth2_authentication",
                enabled=self.environment in ["staging", "prod"],
                description="Autenticação OAuth2 para MCP Server",
                environments=["staging", "prod"],
                metadata={"category": "security", "impact": "high"}
            ),
            FeatureConfig(
                name="enhanced_logging",
                enabled=True,
                description="Sistema de logging estruturado com métricas",
                environments=["dev", "staging", "prod"],
                metadata={"category": "observability", "impact": "medium"}
            ),
            FeatureConfig(
                name="performance_monitoring",
                enabled=True,
                description="Monitoramento de performance e métricas",
                environments=["dev", "staging", "prod"],
                metadata={"category": "observability", "impact": "low"}
            ),
            FeatureConfig(
                name="experimental_features",
                enabled=self.environment == "dev",
                description="Features experimentais em desenvolvimento",
                environments=["dev"],
                rollout_percentage=25,
                metadata={"category": "experimental", "impact": "low"}
            )
        ]

        for feature in default_features:
            self.features[feature.name] = feature

    def _load_from_file(self):
        """Carrega configuração do arquivo JSON."""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            for feature_name, feature_data in config_data.get("features", {}).items():
                if feature_name in self.features:
                    # Atualizar feature existente
                    feature = self.features[feature_name]
                    feature.enabled = feature_data.get("enabled", feature.enabled)
                    feature.rollout_percentage = feature_data.get("rollout_percentage", feature.rollout_percentage)
                    feature.user_whitelist = feature_data.get("user_whitelist", feature.user_whitelist)
                    feature.dependencies = feature_data.get("dependencies", feature.dependencies)
                else:
                    # Criar nova feature
                    self.features[feature_name] = FeatureConfig(
                        name=feature_name,
                        enabled=feature_data.get("enabled", False),
                        description=feature_data.get("description", ""),
                        environments=feature_data.get("environments", ["dev", "staging", "prod"]),
                        rollout_percentage=feature_data.get("rollout_percentage", 0),
                        user_whitelist=feature_data.get("user_whitelist", []),
                        dependencies=feature_data.get("dependencies", []),
                        metadata=feature_data.get("metadata", {})
                    )

            logger.info(f"Configuração carregada do arquivo: {self.config_file}")
        except Exception as e:
            logger.warning(f"Erro ao carregar arquivo de configuração: {e}")

    def _load_from_env(self):
        """Carrega configuração de variáveis de ambiente."""
        for feature_name in self.features:
            env_var = f"FEATURE_{feature_name.upper()}_ENABLED"
            if env_var in os.environ:
                enabled = os.getenv(env_var, "false").lower() in ("true", "1", "yes", "on")
                self.features[feature_name].enabled = enabled
                logger.info(f"Feature {feature_name} configurada via env var: {enabled}")

    def _validate_dependencies(self):
        """Valida dependencies entre features."""
        for feature_name, feature in self.features.items():
            for dependency in feature.dependencies:
                if dependency not in self.features:
                    logger.warning(f"Feature {feature_name} depende de {dependency} que não existe")
                elif not self.features[dependency].enabled:
                    logger.info(f"Feature {feature_name} desabilitada devido à dependência {dependency}")
                    feature.enabled = False

    def is_enabled(self, feature_name: str, user_id: Optional[str] = None) -> bool:
        """
        Verifica se uma feature está habilitada.

        Args:
            feature_name: Nome da feature
            user_id: ID do usuário (para whitelist e percentage)

        Returns:
            True se a feature estiver habilitada
        """
        if feature_name not in self.features:
            logger.warning(f"Feature desconhecida: {feature_name}")
            return False

        feature = self.features[feature_name]

        # Verificar se ambiente está habilitado
        if self.environment not in feature.environments:
            return False

        # Verificar se feature está desabilitada globalmente
        if not feature.enabled:
            return False

        # Verificar dependencies
        for dependency in feature.dependencies:
            if not self.is_enabled(dependency, user_id):
                return False

        # Verificar whitelist
        if feature.user_whitelist and user_id:
            if user_id in feature.user_whitelist:
                return True
            # Se tem whitelist mas usuário não está nela, continuar para percentage

        # Verificar rollout percentage
        if feature.rollout_percentage > 0 and feature.rollout_percentage < 100:
            if user_id:
                # Usar hash do user_id para deterministic rollout
                import hashlib
                hash_value = int(hashlib.md5(f"{feature_name}:{user_id}".encode()).hexdigest()[:8], 16)
                percentage = (hash_value % 100) + 1
                return percentage <= feature.rollout_percentage
            else:
                # Sem user_id, usar rollout simples
                import random
                return random.randint(1, 100) <= feature.rollout_percentage

        # Feature habilitada sem restrições
        return True

    def require_feature(self, feature_name: str, fallback_return=None):
        """
        Decorator que requer que uma feature esteja habilitada.

        Args:
            feature_name: Nome da feature requerida
            fallback_return: Valor a retornar se feature estiver desabilitada
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.is_enabled(feature_name):
                    return func(*args, **kwargs)
                else:
                    logger.info(f"Feature {feature_name} desabilitada, pulando execução de {func.__name__}")
                    return fallback_return
            return wrapper
        return decorator

    def optional_feature(self, feature_name: str, default_behavior: Callable = None):
        """
        Decorator que executa função apenas se feature estiver habilitada,
        senão executa comportamento padrão.

        Args:
            feature_name: Nome da feature
            default_behavior: Função alternativa se feature estiver desabilitada
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.is_enabled(feature_name):
                    return func(*args, **kwargs)
                elif default_behavior:
                    logger.info(f"Feature {feature_name} desabilitada, usando comportamento padrão")
                    return default_behavior(*args, **kwargs)
                else:
                    logger.info(f"Feature {feature_name} desabilitada, nenhuma ação executada")
                    return None
            return wrapper
        return decorator

    def get_enabled_features(self) -> List[str]:
        """Retorna lista de features habilitadas no ambiente atual."""
        return [name for name, feature in self.features.items()
                if self.is_enabled(name)]

    def get_feature_status(self) -> Dict[str, Dict[str, Any]]:
        """Retorna status completo de todas as features."""
        status = {}
        for name, feature in self.features.items():
            status[name] = {
                "enabled": self.is_enabled(name),
                "description": feature.description,
                "environment_enabled": self.environment in feature.environments,
                "rollout_percentage": feature.rollout_percentage,
                "user_whitelist_count": len(feature.user_whitelist),
                "dependencies": feature.dependencies,
                "metadata": feature.metadata
            }
        return status

    def enable_feature(self, feature_name: str, persist: bool = False):
        """Habilita uma feature programaticamente."""
        if feature_name in self.features:
            self.features[feature_name].enabled = True
            logger.info(f"Feature {feature_name} habilitada")

            if persist:
                self._save_to_file()
        else:
            logger.warning(f"Tentativa de habilitar feature desconhecida: {feature_name}")

    def disable_feature(self, feature_name: str, persist: bool = False):
        """Desabilita uma feature programaticamente."""
        if feature_name in self.features:
            self.features[feature_name].enabled = False
            logger.info(f"Feature {feature_name} desabilitada")

            if persist:
                self._save_to_file()
        else:
            logger.warning(f"Tentativa de desabilitar feature desconhecida: {feature_name}")

    def _save_to_file(self):
        """Salva configuração atual no arquivo JSON."""
        try:
            config_data = {"features": {}}
            for name, feature in self.features.items():
                config_data["features"][name] = {
                    "enabled": feature.enabled,
                    "description": feature.description,
                    "environments": feature.environments,
                    "rollout_percentage": feature.rollout_percentage,
                    "user_whitelist": feature.user_whitelist,
                    "dependencies": feature.dependencies,
                    "metadata": feature.metadata
                }

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Configuração salva em: {self.config_file}")
        except Exception as e:
            logger.error(f"Erro ao salvar configuração: {e}")


# Instância global do gerenciador
_feature_manager = None

def get_feature_manager() -> FeatureToggleManager:
    """Retorna instância global do gerenciador de features."""
    global _feature_manager
    if _feature_manager is None:
        _feature_manager = FeatureToggleManager()
    return _feature_manager

# Funções de conveniência
def is_feature_enabled(feature_name: str, user_id: Optional[str] = None) -> bool:
    """Função de conveniência para verificar se feature está habilitada."""
    return get_feature_manager().is_enabled(feature_name, user_id)

def require_feature(feature_name: str, fallback_return=None):
    """Decorator de conveniência que requer feature habilitada."""
    return get_feature_manager().require_feature(feature_name, fallback_return)

def optional_feature(feature_name: str, default_behavior: Callable = None):
    """Decorator de conveniência para features opcionais."""
    return get_feature_manager().optional_feature(feature_name, default_behavior)


if __name__ == "__main__":
    # Exemplo de uso do sistema de feature toggles
    print("🎛️ Testando sistema de Feature Toggles")

    # Criar gerenciador
    manager = FeatureToggleManager()

    # Mostrar status das features
    print("\n📊 Status das Features:")
    for feature_name, status in manager.get_feature_status().items():
        enabled_icon = "✅" if status["enabled"] else "❌"
        print(f"{enabled_icon} {feature_name}: {status['description']}")

    # Testar decorators
    @manager.require_feature("advanced_reasoning")
    def test_reasoning():
        return "🧠 Reasoning avançado executado!"

    @manager.require_feature("prompt_evaluation", fallback_return="⚠️ Avaliação não disponível")
    def test_evaluation():
        return "📊 Avaliação de prompts executada!"

    print(f"\n🧪 Testando funcionalidades:")
    print(f"Reasoning: {test_reasoning()}")
    print(f"Evaluation: {test_evaluation()}")

    # Mostrar features habilitadas
    enabled_features = manager.get_enabled_features()
    print(f"\n🎯 Features habilitadas: {', '.join(enabled_features)}")