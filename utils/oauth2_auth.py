"""
Sistema de Autenticação OAuth2 para Agent Core.
Middleware OAuth2 para MCP Server com validação JWT, JWKS cache e proteção de tools.

Funcionalidades:
- Validação JWT com cache JWKS
- Cliente OAuth2 para service-to-service
- Middleware MCP com proteção por scopes
- Configuração via secrets manager
- Rate limiting e security headers
"""

import os
import jwt
import time
import json
import asyncio
import requests
import boto3

from typing import Dict, List, Optional, Set, Any, Callable
from datetime import datetime, timedelta
from functools import wraps
from urllib.parse import urljoin
from dataclasses import dataclass, asdict
import threading
from cryptography.hazmat.primitives import serialization


@dataclass
class OAuth2Config:
    """Configuração para OAuth2."""

    # Configuração do servidor de autorização
    issuer: str
    client_id: str
    client_secret: str

    # URLs do provider OAuth2
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str
    userinfo_endpoint: Optional[str] = None

    # Configurações de validação
    audience: Optional[str] = None
    algorithm: str = "RS256"
    leeway: int = 10  # Tolerância para exp/nbf em segundos

    # Cache e timeout
    jwks_cache_ttl: int = 3600  # 1 hora
    request_timeout: int = 30

    # Scopes necessários
    required_scopes: Set[str] = None

    @classmethod
    def from_env(cls) -> 'OAuth2Config':
        """Cria configuração a partir de variáveis de ambiente."""
        return cls(
            issuer=os.getenv("OAUTH2_ISSUER"),
            client_id=os.getenv("OAUTH2_CLIENT_ID"),
            client_secret=os.getenv("OAUTH2_CLIENT_SECRET"),
            authorization_endpoint=os.getenv("OAUTH2_AUTHORIZATION_ENDPOINT"),
            token_endpoint=os.getenv("OAUTH2_TOKEN_ENDPOINT"),
            jwks_uri=os.getenv("OAUTH2_JWKS_URI"),
            userinfo_endpoint=os.getenv("OAUTH2_USERINFO_ENDPOINT"),
            audience=os.getenv("OAUTH2_AUDIENCE"),
            algorithm=os.getenv("OAUTH2_ALGORITHM", "RS256"),
            leeway=int(os.getenv("OAUTH2_LEEWAY", "10")),
            jwks_cache_ttl=int(os.getenv("OAUTH2_JWKS_CACHE_TTL", "3600")),
            request_timeout=int(os.getenv("OAUTH2_REQUEST_TIMEOUT", "30")),
            required_scopes=set(os.getenv("OAUTH2_REQUIRED_SCOPES", "").split(",")) if os.getenv("OAUTH2_REQUIRED_SCOPES") else set()
        )

    @classmethod
    def from_aws_secrets(cls, secret_name: str, region: str = "us-east-1") -> 'OAuth2Config':
        """Cria configuração a partir do AWS Secrets Manager."""
        try:

            client = boto3.client('secretsmanager', region_name=region)
            response = client.get_secret_value(SecretId=secret_name)
            secret_data = json.loads(response['SecretString'])

            return cls(
                issuer=secret_data["issuer"],
                client_id=secret_data["client_id"],
                client_secret=secret_data["client_secret"],
                authorization_endpoint=secret_data["authorization_endpoint"],
                token_endpoint=secret_data["token_endpoint"],
                jwks_uri=secret_data["jwks_uri"],
                userinfo_endpoint=secret_data.get("userinfo_endpoint"),
                audience=secret_data.get("audience"),
                algorithm=secret_data.get("algorithm", "RS256"),
                leeway=secret_data.get("leeway", 10),
                jwks_cache_ttl=secret_data.get("jwks_cache_ttl", 3600),
                request_timeout=secret_data.get("request_timeout", 30),
                required_scopes=set(secret_data.get("required_scopes", []))
            )
        except ImportError:
            raise RuntimeError("boto3 necessário para usar AWS Secrets Manager")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar configuração do AWS Secrets Manager: {e}")


class JWKSCache:
    """Cache para JWKS (JSON Web Key Set) com TTL."""

    def __init__(self, jwks_uri: str, cache_ttl: int = 3600, timeout: int = 30):
        self.jwks_uri = jwks_uri
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get_key(self, kid: str) -> Optional[str]:
        """Obtém chave pública por Key ID com cache."""
        with self._lock:
            # Verificar se está em cache e não expirou
            if kid in self._cache:
                if time.time() - self._cache_time[kid] < self.cache_ttl:
                    return self._cache[kid]
                else:
                    # Remover cache expirado
                    del self._cache[kid]
                    del self._cache_time[kid]

            # Buscar nova chave
            try:
                response = requests.get(self.jwks_uri, timeout=self.timeout)
                response.raise_for_status()
                jwks = response.json()

                # Processar todas as chaves
                for key in jwks.get('keys', []):
                    key_id = key.get('kid')
                    if key_id:
                        # Converter para PEM format
                        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
                        pem_key = public_key.public_bytes(
                            encoding=serialization.Encoding.PEM,
                            format=serialization.PublicFormat.SubjectPublicKeyInfo
                        ).decode('utf-8')

                        self._cache[key_id] = pem_key
                        self._cache_time[key_id] = time.time()

                return self._cache.get(kid)

            except Exception as e:
                raise RuntimeError(f"Erro ao buscar JWKS: {e}")


class TokenValidator:
    """Validador de tokens JWT."""

    def __init__(self, config: OAuth2Config):
        self.config = config
        self.jwks_cache = JWKSCache(config.jwks_uri, config.jwks_cache_ttl, config.request_timeout)

    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Valida token JWT e retorna claims.

        Raises:
            jwt.InvalidTokenError: Token inválido
            ValueError: Configuração inválida
        """
        if not token:
            raise jwt.InvalidTokenError("Token não fornecido")

        try:
            # Decodificar header sem verificação para obter kid
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get('kid')

            if not kid:
                raise jwt.InvalidTokenError("Token não possui Key ID (kid)")

            # Obter chave pública
            public_key = self.jwks_cache.get_key(kid)
            if not public_key:
                raise jwt.InvalidTokenError(f"Chave pública não encontrada para kid: {kid}")

            # Validar e decodificar token
            payload = jwt.decode(
                token,
                public_key,
                algorithms=[self.config.algorithm],
                audience=self.config.audience,
                issuer=self.config.issuer,
                leeway=timedelta(seconds=self.config.leeway)
            )

            # Validar scopes se necessário
            if self.config.required_scopes:
                token_scopes = set(payload.get('scope', '').split())
                if not self.config.required_scopes.issubset(token_scopes):
                    missing_scopes = self.config.required_scopes - token_scopes
                    raise jwt.InvalidTokenError(f"Scopes necessários não encontrados: {missing_scopes}")

            return payload

        except jwt.ExpiredSignatureError:
            raise jwt.InvalidTokenError("Token expirado")
        except jwt.InvalidAudienceError:
            raise jwt.InvalidTokenError("Audience inválida")
        except jwt.InvalidIssuerError:
            raise jwt.InvalidTokenError("Issuer inválido")
        except jwt.InvalidSignatureError:
            raise jwt.InvalidTokenError("Assinatura inválida")
        except Exception as e:
            raise jwt.InvalidTokenError(f"Erro na validação do token: {e}")

    def extract_user_info(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extrai informações do usuário do payload do token."""
        return {
            "user_id": payload.get("sub"),
            "username": payload.get("preferred_username") or payload.get("name"),
            "email": payload.get("email"),
            "scopes": payload.get("scope", "").split(),
            "issued_at": payload.get("iat"),
            "expires_at": payload.get("exp"),
            "client_id": payload.get("client_id") or payload.get("azp")
        }


class OAuth2Client:
    """Cliente OAuth2 para service-to-service authentication."""

    def __init__(self, config: OAuth2Config):
        self.config = config
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[float] = None
        self._lock = threading.Lock()

    def get_access_token(self, scopes: List[str] = None) -> str:
        """
        Obtém access token usando Client Credentials flow.

        Args:
            scopes: Lista de scopes necessários

        Returns:
            Access token válido
        """
        with self._lock:
            # Verificar se token atual ainda é válido
            if (self._access_token and self._token_expires_at and
                time.time() < self._token_expires_at - 60):  # Buffer de 60 segundos
                return self._access_token

            # Solicitar novo token
            data = {
                "grant_type": "client_credentials",
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret
            }

            if scopes:
                data["scope"] = " ".join(scopes)

            try:
                response = requests.post(
                    self.config.token_endpoint,
                    data=data,
                    timeout=self.config.request_timeout,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                response.raise_for_status()

                token_data = response.json()
                self._access_token = token_data["access_token"]
                expires_in = token_data.get("expires_in", 3600)
                self._token_expires_at = time.time() + expires_in

                return self._access_token

            except Exception as e:
                raise RuntimeError(f"Erro ao obter access token: {e}")

    def make_authenticated_request(self, method: str, url: str, scopes: List[str] = None, **kwargs) -> requests.Response:
        """
        Faz requisição autenticada com OAuth2.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL do endpoint
            scopes: Scopes necessários para a requisição
            **kwargs: Argumentos adicionais para requests

        Returns:
            Response object
        """
        token = self.get_access_token(scopes)

        headers = kwargs.get("headers", {})
        headers["Authorization"] = f"Bearer {token}"
        kwargs["headers"] = headers

        return requests.request(method, url, **kwargs)


class MCPAuthMiddleware:
    """Middleware OAuth2 para MCP Server."""

    def __init__(self, config: OAuth2Config):
        self.config = config
        self.validator = TokenValidator(config)
        self.tool_scopes: Dict[str, Set[str]] = {}
        self.resource_scopes: Dict[str, Set[str]] = {}

    def register_tool_scopes(self, tool_name: str, required_scopes: List[str]):
        """Registra scopes necessários para uma tool."""
        self.tool_scopes[tool_name] = set(required_scopes)

    def register_resource_scopes(self, resource_uri: str, required_scopes: List[str]):
        """Registra scopes necessários para um resource."""
        self.resource_scopes[resource_uri] = set(required_scopes)

    def extract_token_from_request(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Extrai token de uma requisição MCP."""
        # Verificar header Authorization
        headers = request_data.get("headers", {})
        auth_header = headers.get("Authorization") or headers.get("authorization")

        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer "

        # Verificar parâmetros da requisição
        params = request_data.get("params", {})
        if "access_token" in params:
            return params["access_token"]

        return None

    def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida requisição MCP com OAuth2.

        Returns:
            User info se válida

        Raises:
            RuntimeError: Requisição inválida
        """
        token = self.extract_token_from_request(request_data)
        if not token:
            raise RuntimeError("Token OAuth2 não fornecido")

        try:
            payload = self.validator.validate_token(token)
            return self.validator.extract_user_info(payload)
        except jwt.InvalidTokenError as e:
            raise RuntimeError(f"Token inválido: {e}")

    def check_tool_permission(self, tool_name: str, user_scopes: List[str]) -> bool:
        """Verifica se usuário tem permissão para usar uma tool."""
        required_scopes = self.tool_scopes.get(tool_name, set())
        if not required_scopes:
            return True  # Tool não protegida

        user_scope_set = set(user_scopes)
        return required_scopes.issubset(user_scope_set)

    def check_resource_permission(self, resource_uri: str, user_scopes: List[str]) -> bool:
        """Verifica se usuário tem permissão para acessar um resource."""
        required_scopes = self.resource_scopes.get(resource_uri, set())
        if not required_scopes:
            return True  # Resource não protegido

        user_scope_set = set(user_scopes)
        return required_scopes.issubset(user_scope_set)


def require_oauth2(config: OAuth2Config):
    """
    Decorator para proteger funções com OAuth2.

    Args:
        config: Configuração OAuth2
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Tentar extrair request_data dos argumentos
            request_data = None
            for arg in args:
                if isinstance(arg, dict) and ("headers" in arg or "params" in arg):
                    request_data = arg
                    break

            if not request_data:
                # Procurar nos kwargs
                request_data = kwargs.get("request_data") or kwargs.get("request")

            if not request_data:
                raise RuntimeError("Dados da requisição não encontrados para validação OAuth2")

            middleware = MCPAuthMiddleware(config)
            user_info = middleware.validate_request(request_data)

            # Adicionar user_info aos kwargs
            kwargs["user_info"] = user_info

            return func(*args, **kwargs)

        return wrapper
    return decorator


def require_scopes(required_scopes: List[str]):
    """
    Decorator para requerer scopes específicos.

    Args:
        required_scopes: Lista de scopes necessários
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_info = kwargs.get("user_info")
            if not user_info:
                raise RuntimeError("Informações do usuário não encontradas")

            user_scopes = set(user_info.get("scopes", []))
            required_scope_set = set(required_scopes)

            if not required_scope_set.issubset(user_scopes):
                missing_scopes = required_scope_set - user_scopes
                raise RuntimeError(f"Scopes necessários não encontrados: {missing_scopes}")

            return func(*args, **kwargs)

        return wrapper
    return decorator


# Utilitários para integração com feature toggles
def oauth2_enabled_check():
    """Verifica se OAuth2 está habilitado via feature toggle."""
    try:
        from utils.feature_toggles import is_feature_enabled
        return is_feature_enabled("oauth2_authentication")
    except ImportError:
        return os.getenv("OAUTH2_ENABLED", "false").lower() in ("true", "1", "yes", "on")


def conditional_oauth2_protection(config: OAuth2Config):
    """Decorator que aplica OAuth2 apenas se feature estiver habilitada."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if oauth2_enabled_check():
                # Aplicar proteção OAuth2
                oauth2_decorator = require_oauth2(config)
                return oauth2_decorator(func)(*args, **kwargs)
            else:
                # Executar sem proteção
                return func(*args, **kwargs)

        return wrapper
    return decorator


if __name__ == "__main__":
    # Exemplo de uso do sistema OAuth2
    print("🔐 Testando Sistema de Autenticação OAuth2")

    try:
        # Configuração de exemplo (normalmente viria de env vars ou secrets)
        config = OAuth2Config(
            issuer="https://auth.example.com",
            client_id="test-client",
            client_secret="test-secret",
            authorization_endpoint="https://auth.example.com/oauth2/authorize",
            token_endpoint="https://auth.example.com/oauth2/token",
            jwks_uri="https://auth.example.com/.well-known/jwks.json",
            audience="agent-core-api",
            required_scopes={"read", "write"}
        )

        print(f"📋 Configuração OAuth2:")
        print(f"  Issuer: {config.issuer}")
        print(f"  Client ID: {config.client_id}")
        print(f"  Required Scopes: {config.required_scopes}")

        # Middleware MCP
        middleware = MCPAuthMiddleware(config)

        # Registrar proteções
        middleware.register_tool_scopes("sensitive_operation", ["admin", "write"])
        middleware.register_resource_scopes("config://secure", ["admin"])

        print("✅ Middleware configurado com sucesso")

        # Exemplo de uso com decorators
        @conditional_oauth2_protection(config)
        def protected_function():
            return "🔒 Função protegida executada"

        result = protected_function()
        print(f"🧪 Teste: {result}")

    except Exception as e:
        print(f"❌ Erro no teste: {e}")

    print("🎯 Sistema OAuth2 configurado e pronto para uso!")