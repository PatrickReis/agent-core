"""
Sistema de Enhanced Logging para Agent Core.
Logging estruturado com métricas, tracing distribuído e múltiplos handlers.

Funcionalidades:
- Logs estruturados em JSON
- Context variables para tracing distribuído
- Métricas automáticas (latência, success rate, etc)
- TimingContext para medição automática
- Múltiplos handlers (console, arquivo, métricas)
- Integração com feature toggles
"""

import os
import json
import time
import logging
import uuid
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from functools import wraps
from collections import defaultdict
import threading


class LogLevel:
    """Constantes para níveis de log."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogContext:
    """Context variables para tracing distribuído."""

    _local = threading.local()

    @classmethod
    def set_request_id(cls, request_id: str):
        """Define o ID da requisição atual."""
        cls._local.request_id = request_id

    @classmethod
    def get_request_id(cls) -> Optional[str]:
        """Obtém o ID da requisição atual."""
        return getattr(cls._local, 'request_id', None)

    @classmethod
    def set_session_id(cls, session_id: str):
        """Define o ID da sessão atual."""
        cls._local.session_id = session_id

    @classmethod
    def get_session_id(cls) -> Optional[str]:
        """Obtém o ID da sessão atual."""
        return getattr(cls._local, 'session_id', None)

    @classmethod
    def set_user_id(cls, user_id: str):
        """Define o ID do usuário atual."""
        cls._local.user_id = user_id

    @classmethod
    def get_user_id(cls) -> Optional[str]:
        """Obtém o ID do usuário atual."""
        return getattr(cls._local, 'user_id', None)

    @classmethod
    def set_operation_name(cls, operation_name: str):
        """Define o nome da operação atual."""
        cls._local.operation_name = operation_name

    @classmethod
    def get_operation_name(cls) -> Optional[str]:
        """Obtém o nome da operação atual."""
        return getattr(cls._local, 'operation_name', None)

    @classmethod
    def get_context(cls) -> Dict[str, str]:
        """Obtém todo o contexto atual."""
        context = {}
        if cls.get_request_id():
            context["request_id"] = cls.get_request_id()
        if cls.get_session_id():
            context["session_id"] = cls.get_session_id()
        if cls.get_user_id():
            context["user_id"] = cls.get_user_id()
        if cls.get_operation_name():
            context["operation_name"] = cls.get_operation_name()
        return context

    @classmethod
    def clear(cls):
        """Limpa todo o contexto."""
        for attr in ["request_id", "session_id", "user_id", "operation_name"]:
            if hasattr(cls._local, attr):
                delattr(cls._local, attr)


class MetricsCollector:
    """Coletor de métricas para operações."""

    def __init__(self):
        self.metrics = defaultdict(lambda: {
            "count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_duration": 0.0,
            "min_duration": float('inf'),
            "max_duration": 0.0,
            "last_execution": None
        })
        self._lock = threading.Lock()

    def record_operation(self, operation_name: str, duration: float, success: bool = True):
        """Registra uma operação executada."""
        with self._lock:
            metric = self.metrics[operation_name]
            metric["count"] += 1
            metric["total_duration"] += duration
            metric["min_duration"] = min(metric["min_duration"], duration)
            metric["max_duration"] = max(metric["max_duration"], duration)
            metric["last_execution"] = datetime.now(timezone.utc).isoformat()

            if success:
                metric["success_count"] += 1
            else:
                metric["error_count"] += 1

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Obtém todas as métricas coletadas."""
        with self._lock:
            result = {}
            for operation, metric in self.metrics.items():
                result[operation] = {
                    "count": metric["count"],
                    "success_rate": metric["success_count"] / metric["count"] if metric["count"] > 0 else 0,
                    "error_rate": metric["error_count"] / metric["count"] if metric["count"] > 0 else 0,
                    "avg_duration_ms": (metric["total_duration"] * 1000) / metric["count"] if metric["count"] > 0 else 0,
                    "min_duration_ms": metric["min_duration"] * 1000 if metric["min_duration"] != float('inf') else 0,
                    "max_duration_ms": metric["max_duration"] * 1000,
                    "last_execution": metric["last_execution"]
                }
            return result

    def reset_metrics(self):
        """Reseta todas as métricas."""
        with self._lock:
            self.metrics.clear()


class StructuredFormatter(logging.Formatter):
    """Formatter para logs estruturados em JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Formata o log record como JSON estruturado."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Adicionar context variables
        context = LogContext.get_context()
        if context:
            log_entry["context"] = context

        # Adicionar exception info se presente
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Adicionar campos extras
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname",
                          "filename", "module", "lineno", "funcName", "created", "msecs",
                          "relativeCreated", "thread", "threadName", "processName", "process",
                          "getMessage", "exc_info", "exc_text", "stack_info"]:
                log_entry[key] = value

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Formatter legível para console."""

    def __init__(self):
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )

    def format(self, record: logging.LogRecord) -> str:
        """Formata o log record para console."""
        # Adicionar context ao final da mensagem se presente
        formatted = super().format(record)

        context = LogContext.get_context()
        if context:
            context_str = " | ".join(f"{k}={v}" for k, v in context.items())
            formatted += f" | {context_str}"

        return formatted


class TimingContext:
    """Context manager para medição automática de tempo."""

    def __init__(self, operation_name: str, logger: 'EnhancedLogger', auto_log: bool = True):
        self.operation_name = operation_name
        self.logger = logger
        self.auto_log = auto_log
        self.start_time = None
        self.end_time = None
        self.success = True
        self.error = None

    def __enter__(self):
        self.start_time = time.time()
        LogContext.set_operation_name(self.operation_name)
        if self.auto_log:
            self.logger.info(f"Iniciando operação: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        if exc_type is not None:
            self.success = False
            self.error = str(exc_val)
            if self.auto_log:
                self.logger.error(f"Operação {self.operation_name} falhou: {self.error}",
                                duration=duration)
        else:
            if self.auto_log:
                self.logger.info(f"Operação {self.operation_name} concluída",
                               duration=duration)

        # Registrar métricas
        self.logger.metrics_collector.record_operation(
            self.operation_name, duration, self.success
        )

    def get_duration(self) -> Optional[float]:
        """Obtém a duração da operação."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class EnhancedLogger:
    """
    Logger aprimorado com funcionalidades estruturadas.

    Exemplo de uso:

    logger = EnhancedLogger("mcp_server")

    # Log básico
    logger.info("Servidor iniciado")

    # Com contexto
    with logger.operation_context("process_request"):
        logger.info("Processando requisição")

    # Com timing automático
    with logger.timing("database_query"):
        # Operação que será cronometrada
        pass

    # Métricas
    metrics = logger.get_metrics()
    """

    def __init__(self, name: str, level: str = "INFO", enable_metrics: bool = True):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.metrics_collector = MetricsCollector() if enable_metrics else None

        # Evitar handlers duplicados
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """Configura handlers para console e arquivo."""
        # Handler para console (formato legível)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(ConsoleFormatter())

        # Handler para arquivo (formato JSON estruturado)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Arquivo com timestamp no nome
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = log_dir / f"enhanced_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def _log(self, level: str, message: str, **kwargs):
        """Log interno com contexto adicional."""
        # Adicionar timestamp e contexto aos kwargs
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }

        # Usar o nível apropriado
        log_func = getattr(self.logger, level.lower())
        log_func(message, extra=log_data)

    def debug(self, message: str, **kwargs):
        """Log debug com contexto opcional."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info com contexto opcional."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning com contexto opcional."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error com contexto opcional e exception info."""
        if error:
            kwargs["error_type"] = type(error).__name__
            kwargs["error_message"] = str(error)
        self._log(LogLevel.ERROR, message, exc_info=error is not None, **kwargs)

    def critical(self, message: str, error: Exception = None, **kwargs):
        """Log critical com contexto opcional."""
        if error:
            kwargs["error_type"] = type(error).__name__
            kwargs["error_message"] = str(error)
        self._log(LogLevel.CRITICAL, message, exc_info=error is not None, **kwargs)

    # Métodos especializados
    def operation_start(self, operation_name: str, **kwargs):
        """Inicia uma operação."""
        LogContext.set_operation_name(operation_name)
        self.info(f"🚀 Iniciando: {operation_name}", operation="start", **kwargs)

    def operation_success(self, operation_name: str, duration: float = None, **kwargs):
        """Marca operação como bem-sucedida."""
        log_data = {"operation": "success"}
        if duration is not None:
            log_data["duration_ms"] = duration * 1000

        self.info(f"✅ Concluído: {operation_name}", **log_data, **kwargs)

        if self.metrics_collector and duration is not None:
            self.metrics_collector.record_operation(operation_name, duration, True)

    def operation_error(self, operation_name: str, error: Exception, duration: float = None, **kwargs):
        """Marca operação como falhada."""
        log_data = {"operation": "error"}
        if duration is not None:
            log_data["duration_ms"] = duration * 1000

        self.error(f"❌ Falhou: {operation_name}", error=error, **log_data, **kwargs)

        if self.metrics_collector and duration is not None:
            self.metrics_collector.record_operation(operation_name, duration, False)

    def agent_decision(self, decision: str, reasoning: str = None, **kwargs):
        """Log de decisão do agente."""
        log_data = {"decision_type": "agent", "decision": decision}
        if reasoning:
            log_data["reasoning"] = reasoning
        self.info(f"🤖 Decisão do agente: {decision}", **log_data, **kwargs)

    def tool_execution(self, tool_name: str, args: Dict[str, Any] = None, **kwargs):
        """Log de execução de ferramenta."""
        log_data = {"tool_name": tool_name}
        if args:
            log_data["tool_args"] = args
        self.info(f"🛠️ Executando tool: {tool_name}", **log_data, **kwargs)

    def performance_metric(self, metric_name: str, value: float, unit: str = "ms", **kwargs):
        """Log de métrica de performance."""
        self.info(f"📊 Métrica: {metric_name}",
                 metric_name=metric_name, metric_value=value, metric_unit=unit, **kwargs)

    @contextmanager
    def timing(self, operation_name: str, auto_log: bool = True):
        """Context manager para timing automático."""
        timing_ctx = TimingContext(operation_name, self, auto_log)
        with timing_ctx:
            yield timing_ctx

    @contextmanager
    def operation_context(self, operation_name: str, **context_vars):
        """Context manager para operação com contexto."""
        # Salvar contexto anterior
        old_operation = LogContext.get_operation_name()

        try:
            # Definir novo contexto
            LogContext.set_operation_name(operation_name)
            for key, value in context_vars.items():
                setattr(LogContext._local, key, value)

            self.operation_start(operation_name, **context_vars)
            start_time = time.time()

            yield

            # Sucesso
            duration = time.time() - start_time
            self.operation_success(operation_name, duration)

        except Exception as e:
            # Erro
            duration = time.time() - start_time if 'start_time' in locals() else None
            self.operation_error(operation_name, e, duration)
            raise

        finally:
            # Restaurar contexto anterior
            if old_operation:
                LogContext.set_operation_name(old_operation)
            else:
                LogContext.clear()

    def log_with_feature_toggle(self, feature_name: str, message: str, level: str = "info", **kwargs):
        """Log condicional baseado em feature toggle."""
        try:
            from utils.feature_toggles import is_feature_enabled
            if is_feature_enabled(feature_name):
                getattr(self, level)(message, feature=feature_name, **kwargs)
        except ImportError:
            # Se feature toggles não estiver disponível, sempre logar
            getattr(self, level)(message, **kwargs)

    def get_metrics(self) -> Dict[str, Any]:
        """Obtém métricas coletadas."""
        if self.metrics_collector:
            return self.metrics_collector.get_metrics()
        return {}

    def reset_metrics(self):
        """Reseta métricas coletadas."""
        if self.metrics_collector:
            self.metrics_collector.reset_metrics()


# Factory function para loggers especializados
def get_enhanced_logger(name: str, level: str = "INFO", enable_metrics: bool = True) -> EnhancedLogger:
    """Factory function para criar loggers aprimorados."""
    return EnhancedLogger(name, level, enable_metrics)


# Decorator para timing automático
def timed_operation(operation_name: str = None, logger: EnhancedLogger = None):
    """Decorator para cronometrar operações automaticamente."""
    def decorator(func: Callable) -> Callable:
        nonlocal operation_name, logger

        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"

        if logger is None:
            logger = get_enhanced_logger("timed_operations")

        @wraps(func)
        def wrapper(*args, **kwargs):
            with logger.timing(operation_name):
                return func(*args, **kwargs)

        return wrapper
    return decorator


# Instâncias padrão para compatibilidade
_default_loggers = {}

def get_logger(component: str, level: str = "INFO") -> EnhancedLogger:
    """Função de compatibilidade que retorna enhanced logger."""
    if component not in _default_loggers:
        _default_loggers[component] = get_enhanced_logger(f"agent_core.{component}", level)
    return _default_loggers[component]


if __name__ == "__main__":
    # Exemplo de uso do Enhanced Logging
    print("📝 Testando Enhanced Logging System")

    # Criar logger
    logger = get_enhanced_logger("test")

    # Definir contexto
    LogContext.set_request_id("req-12345")
    LogContext.set_session_id("session-67890")

    # Logs básicos
    logger.info("Sistema iniciado")
    logger.debug("Informação de debug", component="test", version="1.0.0")

    # Operação com timing
    with logger.timing("test_operation"):
        time.sleep(0.1)  # Simular trabalho
        logger.info("Trabalho realizado")

    # Context de operação
    try:
        with logger.operation_context("complex_task", task_id="task-001"):
            logger.info("Executando tarefa complexa")
            # Simular erro
            raise ValueError("Erro simulado")
    except ValueError:
        pass  # Erro já foi logado automaticamente

    # Mostrar métricas
    metrics = logger.get_metrics()
    print(f"\n📊 Métricas coletadas:")
    for operation, data in metrics.items():
        print(f"  {operation}: {data['count']} execuções, {data['success_rate']:.1%} sucesso")