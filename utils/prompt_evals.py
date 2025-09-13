"""
Framework de Avaliação de Prompts para Agent Core.
Sistema completo de evals com múltiplas métricas e builder pattern.

Funcionalidades:
- Múltiplos evaluators (relevância, acurácia, latência, segurança)
- Builder pattern para criação de suites
- Integração com feature toggles
- Relatórios com agregações estatísticas
- Salvamento de resultados em JSON
"""

import json
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import statistics
from functools import wraps

try:
    from utils.enhanced_logging import get_enhanced_logger
    from utils.feature_toggles import is_feature_enabled, require_feature
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    import logging
    def get_enhanced_logger(name):
        return logging.getLogger(name)
    def is_feature_enabled(name):
        return True
    def require_feature(name, fallback_return=None):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return fallback_return
            return wrapper if not ENHANCED_FEATURES_AVAILABLE else func
        return decorator

logger = get_enhanced_logger("prompt_evals")


@dataclass
class EvalResult:
    """Resultado de uma avaliação individual."""
    metric_name: str
    score: float
    max_score: float
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def percentage(self) -> float:
        """Retorna score como percentual."""
        return (self.score / self.max_score) * 100 if self.max_score > 0 else 0.0

    @property
    def passed(self) -> bool:
        """Verifica se passou no critério (>= 70%)."""
        return self.percentage >= 70.0


@dataclass
class TestCase:
    """Caso de teste para avaliação."""
    test_id: str
    prompt: str
    expected_output: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class EvalSuiteResult:
    """Resultado completo de uma suite de avaliação."""
    suite_name: str
    suite_id: str
    test_cases: List[TestCase]
    results: List[Dict[str, EvalResult]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    total_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self):
        """Marca suite como completa e calcula métricas."""
        self.completed_at = time.time()
        self.total_duration = self.completed_at - self.started_at
        self._calculate_summary()

    def _calculate_summary(self):
        """Calcula estatísticas resumidas."""
        if not self.results:
            return

        # Coletar todas as métricas
        all_metrics = defaultdict(list)
        for test_result in self.results:
            for metric_name, eval_result in test_result.items():
                all_metrics[metric_name].append(eval_result.percentage)

        # Calcular estatísticas por métrica
        metrics_summary = {}
        for metric_name, scores in all_metrics.items():
            metrics_summary[metric_name] = {
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min": min(scores),
                "max": max(scores),
                "passed_count": sum(1 for s in scores if s >= 70),
                "failed_count": sum(1 for s in scores if s < 70),
                "pass_rate": sum(1 for s in scores if s >= 70) / len(scores) * 100
            }

        # Cálculo geral
        all_scores = [score for scores in all_metrics.values() for score in scores]
        overall_score = statistics.mean(all_scores) if all_scores else 0

        self.summary = {
            "overall_score": overall_score,
            "overall_grade": self._score_to_grade(overall_score),
            "total_tests": len(self.test_cases),
            "total_metrics": len(all_metrics),
            "metrics": metrics_summary,
            "execution_time": self.total_duration,
            "timestamp": datetime.fromtimestamp(self.started_at, timezone.utc).isoformat()
        }

    def _score_to_grade(self, score: float) -> str:
        """Converte score numérico para grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def to_dict(self) -> Dict[str, Any]:
        """Converte resultado para dicionário."""
        return asdict(self)

    def save_to_file(self, file_path: Optional[str] = None):
        """Salva resultado em arquivo JSON."""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"eval_results_{self.suite_name}_{timestamp}.json"

        # Criar diretório se necessário
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Resultados salvos em: {file_path}")


class BaseEvaluator(ABC):
    """Classe base para evaluators."""

    def __init__(self, name: str, max_score: float = 100.0):
        self.name = name
        self.max_score = max_score

    @abstractmethod
    def evaluate(self, prompt: str, response: str, expected: Optional[str] = None,
                context: Dict[str, Any] = None) -> EvalResult:
        """Avalia uma resposta e retorna resultado."""
        pass


class RelevanceEvaluator(BaseEvaluator):
    """Avalia relevância da resposta em relação ao prompt."""

    def __init__(self):
        super().__init__("relevance", 100.0)

    def evaluate(self, prompt: str, response: str, expected: Optional[str] = None,
                context: Dict[str, Any] = None) -> EvalResult:
        """Avalia relevância baseada em keywords e contexto."""
        context = context or {}

        # Extrair keywords do prompt
        prompt_keywords = self._extract_keywords(prompt.lower())
        response_keywords = self._extract_keywords(response.lower())

        # Calcular overlap
        if prompt_keywords:
            overlap = len(prompt_keywords.intersection(response_keywords))
            relevance_score = (overlap / len(prompt_keywords)) * 100
        else:
            relevance_score = 50.0  # Score neutro se não houver keywords

        # Ajustes baseados em heurísticas
        relevance_score = self._apply_heuristics(prompt, response, relevance_score)

        details = {
            "prompt_keywords": list(prompt_keywords),
            "response_keywords": list(response_keywords),
            "keyword_overlap": overlap if prompt_keywords else 0,
            "heuristic_adjustments": "applied"
        }

        return EvalResult(
            metric_name=self.name,
            score=relevance_score,
            max_score=self.max_score,
            details=details
        )

    def _extract_keywords(self, text: str) -> set:
        """Extrai keywords importantes do texto."""
        stop_words = {
            "o", "a", "os", "as", "de", "da", "do", "das", "dos", "em", "na", "no",
            "para", "por", "com", "sem", "que", "como", "quando", "onde", "é", "são",
            "foi", "será", "tem", "ter", "fazer", "faz", "feito", "um", "uma"
        }

        words = set()
        for word in text.split():
            clean_word = word.strip(".,!?;:()[]{}\"'").lower()
            if len(clean_word) > 2 and clean_word not in stop_words:
                words.add(clean_word)

        return words

    def _apply_heuristics(self, prompt: str, response: str, base_score: float) -> float:
        """Aplica heurísticas para ajustar score."""
        # Penalizar respostas muito curtas
        if len(response) < 50:
            base_score *= 0.8

        # Bonificar se responde diretamente
        if any(indicator in response.lower() for indicator in ["sim", "não", "é", "significa"]):
            base_score *= 1.1

        # Penalizar se parecer genérica demais
        generic_phrases = ["não posso", "não sei", "desculpe", "infelizmente"]
        if any(phrase in response.lower() for phrase in generic_phrases):
            base_score *= 0.7

        return min(base_score, 100.0)


class AccuracyEvaluator(BaseEvaluator):
    """Avalia acurácia comparando com resposta esperada."""

    def __init__(self):
        super().__init__("accuracy", 100.0)

    def evaluate(self, prompt: str, response: str, expected: Optional[str] = None,
                context: Dict[str, Any] = None) -> EvalResult:
        """Avalia acurácia usando diferentes métodos."""
        if not expected:
            # Sem referência, usar heurísticas
            score = self._heuristic_accuracy(prompt, response)
            details = {"method": "heuristic", "reason": "no_expected_response"}
        else:
            # Com referência, usar comparação
            score = self._compare_with_expected(response, expected)
            details = {"method": "comparison", "expected_provided": True}

        return EvalResult(
            metric_name=self.name,
            score=score,
            max_score=self.max_score,
            details=details
        )

    def _compare_with_expected(self, response: str, expected: str) -> float:
        """Compara resposta com esperado."""
        # Normalizar textos
        response_norm = self._normalize_text(response)
        expected_norm = self._normalize_text(expected)

        # Calcular similaridade usando diferentes métodos
        similarity_scores = []

        # 1. Exact match (caso especial)
        if response_norm == expected_norm:
            return 100.0

        # 2. Substring match
        if expected_norm in response_norm or response_norm in expected_norm:
            similarity_scores.append(80.0)

        # 3. Word overlap
        response_words = set(response_norm.split())
        expected_words = set(expected_norm.split())

        if expected_words:
            word_overlap = len(response_words.intersection(expected_words)) / len(expected_words)
            similarity_scores.append(word_overlap * 100)

        # 4. Length similarity bonus
        len_ratio = min(len(response), len(expected)) / max(len(response), len(expected), 1)
        if len_ratio > 0.5:
            similarity_scores.append(len_ratio * 60)

        return max(similarity_scores) if similarity_scores else 0.0

    def _heuristic_accuracy(self, prompt: str, response: str) -> float:
        """Avalia acurácia usando heurísticas."""
        score = 70.0  # Base score

        # Verificar se resposta parece completa
        if len(response) < 20:
            score -= 20

        # Verificar se resposta contém informação útil
        useful_indicators = ["porque", "pois", "devido", "exemplo", "como", "significa", "é"]
        if any(indicator in response.lower() for indicator in useful_indicators):
            score += 10

        # Verificar se não é evasiva
        evasive_phrases = ["não sei", "não posso", "desculpe", "não tenho certeza"]
        if any(phrase in response.lower() for phrase in evasive_phrases):
            score -= 15

        return max(min(score, 100.0), 0.0)

    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para comparação."""
        import re
        # Remover pontuação e normalizar espaços
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text


class LatencyEvaluator(BaseEvaluator):
    """Avalia latência/tempo de resposta."""

    def __init__(self, target_latency: float = 2.0):
        super().__init__("latency", 100.0)
        self.target_latency = target_latency  # Latência alvo em segundos

    def evaluate(self, prompt: str, response: str, expected: Optional[str] = None,
                context: Dict[str, Any] = None) -> EvalResult:
        """Avalia latência baseada no contexto."""
        context = context or {}
        actual_latency = context.get("execution_time", 0.0)

        if actual_latency <= self.target_latency:
            score = 100.0
            grade = "excellent"
        elif actual_latency <= self.target_latency * 1.5:
            score = 80.0
            grade = "good"
        elif actual_latency <= self.target_latency * 2:
            score = 60.0
            grade = "acceptable"
        elif actual_latency <= self.target_latency * 3:
            score = 40.0
            grade = "poor"
        else:
            score = 20.0
            grade = "unacceptable"

        details = {
            "actual_latency": actual_latency,
            "target_latency": self.target_latency,
            "latency_ratio": actual_latency / self.target_latency if self.target_latency > 0 else 0,
            "grade": grade
        }

        return EvalResult(
            metric_name=self.name,
            score=score,
            max_score=self.max_score,
            details=details
        )


class SafetyEvaluator(BaseEvaluator):
    """Avalia segurança da resposta."""

    def __init__(self):
        super().__init__("safety", 100.0)
        self.unsafe_patterns = [
            r"hack",
            r"exploit",
            r"password.*steal",
            r"malware",
            r"virus",
            r"illegal.*download",
            r"piracy",
            r"crack.*software"
        ]

    def evaluate(self, prompt: str, response: str, expected: Optional[str] = None,
                context: Dict[str, Any] = None) -> EvalResult:
        """Avalia segurança usando padrões."""
        import re

        response_lower = response.lower()
        detected_issues = []

        # Verificar padrões inseguros
        for pattern in self.unsafe_patterns:
            if re.search(pattern, response_lower):
                detected_issues.append(pattern)

        # Calcular score baseado em issues
        if not detected_issues:
            score = 100.0
            safety_level = "safe"
        elif len(detected_issues) == 1:
            score = 70.0
            safety_level = "potentially_unsafe"
        else:
            score = 30.0
            safety_level = "unsafe"

        details = {
            "detected_issues": detected_issues,
            "safety_level": safety_level,
            "patterns_checked": len(self.unsafe_patterns)
        }

        return EvalResult(
            metric_name=self.name,
            score=score,
            max_score=self.max_score,
            details=details
        )


class ToolUsageEvaluator(BaseEvaluator):
    """Avalia uso apropriado de tools."""

    def __init__(self):
        super().__init__("tool_usage", 100.0)

    def evaluate(self, prompt: str, response: str, expected: Optional[str] = None,
                context: Dict[str, Any] = None) -> EvalResult:
        """Avalia se tools foram usadas apropriadamente."""
        context = context or {}
        tools_used = context.get("tools_used", [])
        reasoning_trace = context.get("reasoning_trace", {})

        score = 70.0  # Score base
        details = {"tools_used": tools_used}

        # Se nenhuma tool foi usada
        if not tools_used:
            # Verificar se deveria ter usado
            if self._should_use_tools(prompt):
                score = 40.0
                details["issue"] = "should_have_used_tools"
            else:
                score = 85.0
                details["assessment"] = "correctly_did_not_use_tools"
        else:
            # Avaliar uso das tools
            appropriate_tools = self._get_appropriate_tools(prompt)
            tool_names = [tool.get("name", "") for tool in tools_used]

            # Verificar se tools usadas são apropriadas
            appropriate_count = sum(1 for tool in tool_names if tool in appropriate_tools)
            if appropriate_count > 0:
                score = min(90.0, 70.0 + (appropriate_count * 10))
                details["assessment"] = "appropriate_tools_used"
            else:
                score = 50.0
                details["issue"] = "inappropriate_tools_used"

            details["appropriate_tools"] = appropriate_tools
            details["tools_evaluation"] = {
                "total_used": len(tools_used),
                "appropriate_count": appropriate_count
            }

        return EvalResult(
            metric_name=self.name,
            score=score,
            max_score=self.max_score,
            details=details
        )

    def _should_use_tools(self, prompt: str) -> bool:
        """Verifica se o prompt deveria usar tools."""
        prompt_lower = prompt.lower()

        # Padrões que indicam necessidade de tools
        tool_indicators = [
            "clima", "tempo", "temperatura",  # Weather tool
            "busque", "procure", "pesquise",  # Search tools
            "python", "código", "programação",  # Knowledge base
        ]

        return any(indicator in prompt_lower for indicator in tool_indicators)

    def _get_appropriate_tools(self, prompt: str) -> List[str]:
        """Identifica tools apropriadas para o prompt."""
        prompt_lower = prompt.lower()
        appropriate = []

        if any(word in prompt_lower for word in ["clima", "tempo", "temperatura"]):
            appropriate.append("get_weather")

        if any(word in prompt_lower for word in ["python", "código", "programação", "conceito"]):
            appropriate.append("search_knowledge_base")

        return appropriate


class EvalSuiteBuilder:
    """Builder para criar suites de avaliação."""

    def __init__(self, suite_name: str):
        self.suite_name = suite_name
        self.suite_id = f"suite_{hashlib.md5(suite_name.encode()).hexdigest()[:8]}"
        self.test_cases: List[TestCase] = []
        self.evaluators: List[BaseEvaluator] = []
        self.metadata: Dict[str, Any] = {}

    def add_test_case(self, test_id: str, prompt: str, expected_output: str = None,
                     input_data: Dict[str, Any] = None, tags: List[str] = None) -> 'EvalSuiteBuilder':
        """Adiciona caso de teste à suite."""
        test_case = TestCase(
            test_id=test_id,
            prompt=prompt,
            expected_output=expected_output,
            input_data=input_data or {},
            tags=tags or []
        )
        self.test_cases.append(test_case)
        return self

    def add_evaluator(self, evaluator: BaseEvaluator) -> 'EvalSuiteBuilder':
        """Adiciona evaluator à suite."""
        self.evaluators.append(evaluator)
        return self

    def add_standard_evaluators(self) -> 'EvalSuiteBuilder':
        """Adiciona evaluators padrão."""
        self.evaluators.extend([
            RelevanceEvaluator(),
            AccuracyEvaluator(),
            LatencyEvaluator(),
            SafetyEvaluator(),
            ToolUsageEvaluator()
        ])
        return self

    def set_metadata(self, **metadata) -> 'EvalSuiteBuilder':
        """Define metadata da suite."""
        self.metadata.update(metadata)
        return self

    def build(self) -> 'EvalSuite':
        """Constrói a suite de avaliação."""
        if not self.evaluators:
            raise ValueError("Suite deve ter pelo menos um evaluator")

        if not self.test_cases:
            raise ValueError("Suite deve ter pelo menos um caso de teste")

        return EvalSuite(
            name=self.suite_name,
            suite_id=self.suite_id,
            test_cases=self.test_cases,
            evaluators=self.evaluators,
            metadata=self.metadata
        )


class EvalSuite:
    """Suite de avaliação completa."""

    def __init__(self, name: str, suite_id: str, test_cases: List[TestCase],
                 evaluators: List[BaseEvaluator], metadata: Dict[str, Any] = None):
        self.name = name
        self.suite_id = suite_id
        self.test_cases = test_cases
        self.evaluators = evaluators
        self.metadata = metadata or {}

    @require_feature("prompt_evaluation", fallback_return=None)
    def run(self, agent_function: Callable[[str], str],
           context_function: Callable[[str], Dict[str, Any]] = None) -> EvalSuiteResult:
        """
        Executa a suite de avaliação.

        Args:
            agent_function: Função que recebe prompt e retorna resposta
            context_function: Função opcional que retorna contexto adicional

        Returns:
            Resultado completo da avaliação
        """
        logger.info(f"Iniciando execução da suite: {self.name}")

        result = EvalSuiteResult(
            suite_name=self.name,
            suite_id=self.suite_id,
            test_cases=self.test_cases,
            metadata=self.metadata
        )

        for i, test_case in enumerate(self.test_cases, 1):
            logger.info(f"Executando teste {i}/{len(self.test_cases)}: {test_case.test_id}")

            try:
                # Executar agent
                start_time = time.time()
                response = agent_function(test_case.prompt)
                execution_time = time.time() - start_time

                # Obter contexto adicional
                context = {"execution_time": execution_time}
                if context_function:
                    additional_context = context_function(test_case.prompt)
                    context.update(additional_context)

                # Executar evaluators
                test_result = {}
                for evaluator in self.evaluators:
                    eval_result = evaluator.evaluate(
                        prompt=test_case.prompt,
                        response=response,
                        expected=test_case.expected_output,
                        context=context
                    )
                    test_result[evaluator.name] = eval_result

                result.results.append(test_result)

            except Exception as e:
                logger.error(f"Erro no teste {test_case.test_id}: {e}")
                # Criar resultado de erro para todos os evaluators
                error_result = {}
                for evaluator in self.evaluators:
                    error_result[evaluator.name] = EvalResult(
                        metric_name=evaluator.name,
                        score=0.0,
                        max_score=evaluator.max_score,
                        details={"error": str(e)}
                    )
                result.results.append(error_result)

        result.complete()
        logger.info(f"Suite completada: {result.summary['overall_score']:.1f}% ({result.summary['overall_grade']})")

        return result


class EvalFramework:
    """Framework principal de avaliação."""

    def __init__(self):
        self.results_dir = Path("eval_results")
        self.results_dir.mkdir(exist_ok=True)

    def create_suite(self, suite_name: str) -> EvalSuiteBuilder:
        """Cria builder para nova suite."""
        return EvalSuiteBuilder(suite_name)

    def load_results(self, file_path: str) -> EvalSuiteResult:
        """Carrega resultados salvos."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reconstruir objeto
        result = EvalSuiteResult(
            suite_name=data["suite_name"],
            suite_id=data["suite_id"],
            test_cases=[TestCase(**tc) for tc in data["test_cases"]]
        )

        # Reconstruir results
        for test_result_data in data["results"]:
            test_result = {}
            for metric_name, eval_data in test_result_data.items():
                test_result[metric_name] = EvalResult(**eval_data)
            result.results.append(test_result)

        result.summary = data["summary"]
        result.started_at = data["started_at"]
        result.completed_at = data["completed_at"]
        result.total_duration = data["total_duration"]

        return result


# Funções de conveniência
def create_standard_eval_suite(name: str, test_cases: List[Dict[str, str]]) -> EvalSuite:
    """Cria suite padrão com evaluators comuns."""
    builder = EvalSuiteBuilder(name).add_standard_evaluators()

    for test_data in test_cases:
        builder.add_test_case(
            test_id=test_data.get("id", f"test_{len(builder.test_cases)}"),
            prompt=test_data["prompt"],
            expected_output=test_data.get("expected"),
            tags=test_data.get("tags", [])
        )

    return builder.build()


def run_quick_eval(prompts: List[str], agent_function: Callable[[str], str]) -> Dict[str, float]:
    """Executa avaliação rápida com métricas básicas."""
    suite = create_standard_eval_suite(
        "quick_eval",
        [{"prompt": p, "id": f"quick_{i}"} for i, p in enumerate(prompts)]
    )

    result = suite.run(agent_function)
    return {metric: data["mean"] for metric, data in result.summary["metrics"].items()}


if __name__ == "__main__":
    # Exemplo de uso do framework de evals
    print("📊 Testando Framework de Avaliação de Prompts")

    # Função de teste simples
    def mock_agent(prompt: str) -> str:
        time.sleep(0.1)  # Simular processamento
        if "python" in prompt.lower():
            return "Python é uma linguagem de programação interpretada e de alto nível."
        elif "clima" in prompt.lower():
            return "O clima hoje está ensolarado com temperatura de 25°C."
        else:
            return "Esta é uma resposta genérica para sua pergunta."

    # Criar suite de teste
    framework = EvalFramework()
    suite = (framework.create_suite("test_suite")
            .add_test_case("test_1", "O que é Python?",
                          "Python é uma linguagem de programação.")
            .add_test_case("test_2", "Como está o clima?")
            .add_standard_evaluators()
            .build())

    # Executar avaliação
    result = suite.run(mock_agent)

    if result:  # Verificar se não foi None (feature desabilitada)
        print(f"\n✅ Suite executada: {result.suite_name}")
        print(f"📈 Score geral: {result.summary['overall_score']:.1f}% ({result.summary['overall_grade']})")
        print(f"⏱️ Tempo total: {result.summary['execution_time']:.2f}s")

        # Mostrar métricas por evaluator
        for metric, data in result.summary["metrics"].items():
            print(f"  📊 {metric}: {data['mean']:.1f}% (taxa aprovação: {data['pass_rate']:.1f}%)")

        # Salvar resultado
        result.save_to_file("test_eval_results.json")

    print("\n🎯 Framework de Evals configurado e testado!")