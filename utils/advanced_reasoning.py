"""
Sistema de Reasoning Avançado para Agent Core.
Reasoning multi-step com análise, planejamento, execução e reflexão.

Funcionalidades:
- Reasoning multi-step estruturado
- Reflexão automática e adaptação dinâmica
- Traces completos de raciocínio
- Orquestração inteligente baseada na complexidade
- Integração com feature toggles e enhanced logging
"""

import json
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod

try:
    from utils.enhanced_logging import get_enhanced_logger, LogContext
    from utils.feature_toggles import is_feature_enabled, require_feature
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    # Fallback para compatibilidade
    ENHANCED_FEATURES_AVAILABLE = False
    import logging
    def get_enhanced_logger(name):
        return logging.getLogger(name)
    def is_feature_enabled(name):
        return True
    def require_feature(name, fallback=None):
        def decorator(func):
            return func
        return decorator

logger = get_enhanced_logger("advanced_reasoning")


class ReasoningStepType(Enum):
    """Tipos de steps do reasoning."""
    ANALYZE = "analyze"
    PLAN = "plan"
    EXECUTE = "execute"
    REFLECT = "reflect"
    DECIDE = "decide"
    GATHER_INFO = "gather_info"
    SYNTHESIZE = "synthesize"


class ReasoningComplexity(Enum):
    """Níveis de complexidade de reasoning."""
    SIMPLE = "simple"        # Resposta direta
    MODERATE = "moderate"    # 2-3 steps
    COMPLEX = "complex"      # 4-6 steps
    ADVANCED = "advanced"    # 7+ steps com reflexão


@dataclass
class ReasoningStep:
    """Representa um step individual do reasoning."""
    step_id: str
    step_type: ReasoningStepType
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReasoningTrace:
    """Trace completo do processo de reasoning."""
    trace_id: str
    question: str
    complexity: ReasoningComplexity
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    confidence: float = 0.0
    total_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def add_step(self, step: ReasoningStep):
        """Adiciona step ao trace."""
        self.steps.append(step)
        logger.info(f"Step adicionado ao reasoning: {step.step_type.value} - {step.description}")

    def complete(self, final_answer: str, confidence: float = 1.0):
        """Completa o trace de reasoning."""
        self.completed_at = time.time()
        self.total_time = self.completed_at - self.started_at
        self.final_answer = final_answer
        self.confidence = confidence
        self.success = True
        logger.info(f"Reasoning trace completado: {len(self.steps)} steps, {self.total_time:.2f}s")

    def fail(self, error_message: str):
        """Marca o trace como falhado."""
        self.completed_at = time.time()
        self.total_time = self.completed_at - self.started_at
        self.success = False
        self.error_message = error_message
        logger.error(f"Reasoning trace falhou: {error_message}")

    def to_dict(self) -> Dict[str, Any]:
        """Converte trace para dicionário."""
        return asdict(self)


class ReasoningAnalyzer:
    """Analisador de complexidade de perguntas."""

    def __init__(self):
        self.complexity_keywords = {
            ReasoningComplexity.SIMPLE: [
                "o que é", "qual é", "quando", "onde", "quem",
                "definição", "significado", "clima"
            ],
            ReasoningComplexity.MODERATE: [
                "como", "por que", "explique", "compare",
                "diferença", "vantagens", "desvantagens"
            ],
            ReasoningComplexity.COMPLEX: [
                "analise", "avalie", "implemente", "desenvolva",
                "otimize", "integre", "arquitetura"
            ],
            ReasoningComplexity.ADVANCED: [
                "estratégia", "roadmap", "múltiplos", "complexo",
                "sistema completo", "orquestração", "multi-step"
            ]
        }

    def analyze_complexity(self, question: str) -> ReasoningComplexity:
        """Analisa a complexidade de uma pergunta."""
        question_lower = question.lower()

        # Contar matches por nível
        complexity_scores = {complexity: 0 for complexity in ReasoningComplexity}

        for complexity, keywords in self.complexity_keywords.items():
            for keyword in keywords:
                if keyword in question_lower:
                    complexity_scores[complexity] += 1

        # Calcular score total
        if complexity_scores[ReasoningComplexity.ADVANCED] > 0:
            return ReasoningComplexity.ADVANCED
        elif complexity_scores[ReasoningComplexity.COMPLEX] > 0:
            return ReasoningComplexity.COMPLEX
        elif complexity_scores[ReasoningComplexity.MODERATE] > 0:
            return ReasoningComplexity.MODERATE
        else:
            return ReasoningComplexity.SIMPLE

    def suggest_steps(self, question: str, complexity: ReasoningComplexity) -> List[ReasoningStepType]:
        """Sugere steps baseado na complexidade."""
        step_templates = {
            ReasoningComplexity.SIMPLE: [
                ReasoningStepType.ANALYZE,
                ReasoningStepType.EXECUTE
            ],
            ReasoningComplexity.MODERATE: [
                ReasoningStepType.ANALYZE,
                ReasoningStepType.PLAN,
                ReasoningStepType.EXECUTE,
                ReasoningStepType.SYNTHESIZE
            ],
            ReasoningComplexity.COMPLEX: [
                ReasoningStepType.ANALYZE,
                ReasoningStepType.GATHER_INFO,
                ReasoningStepType.PLAN,
                ReasoningStepType.EXECUTE,
                ReasoningStepType.REFLECT,
                ReasoningStepType.SYNTHESIZE
            ],
            ReasoningComplexity.ADVANCED: [
                ReasoningStepType.ANALYZE,
                ReasoningStepType.GATHER_INFO,
                ReasoningStepType.PLAN,
                ReasoningStepType.DECIDE,
                ReasoningStepType.EXECUTE,
                ReasoningStepType.REFLECT,
                ReasoningStepType.SYNTHESIZE
            ]
        }

        return step_templates.get(complexity, [ReasoningStepType.EXECUTE])


class ReasoningExecutor(ABC):
    """Interface para executores de reasoning steps."""

    @abstractmethod
    def execute_step(self, step_type: ReasoningStepType, input_data: Dict[str, Any],
                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Executa um step específico de reasoning."""
        pass


class DefaultReasoningExecutor(ReasoningExecutor):
    """Executor padrão de reasoning steps."""

    def __init__(self, llm_provider=None, tools: List[Any] = None):
        self.llm_provider = llm_provider
        self.tools = tools or []

    def execute_step(self, step_type: ReasoningStepType, input_data: Dict[str, Any],
                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Executa um step baseado no tipo."""
        if step_type == ReasoningStepType.ANALYZE:
            return self._analyze_step(input_data, context)
        elif step_type == ReasoningStepType.PLAN:
            return self._plan_step(input_data, context)
        elif step_type == ReasoningStepType.EXECUTE:
            return self._execute_step(input_data, context)
        elif step_type == ReasoningStepType.REFLECT:
            return self._reflect_step(input_data, context)
        elif step_type == ReasoningStepType.DECIDE:
            return self._decide_step(input_data, context)
        elif step_type == ReasoningStepType.GATHER_INFO:
            return self._gather_info_step(input_data, context)
        elif step_type == ReasoningStepType.SYNTHESIZE:
            return self._synthesize_step(input_data, context)
        else:
            return {"error": f"Step type não suportado: {step_type}"}

    def _analyze_step(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Step de análise inicial."""
        question = input_data.get("question", "")

        analysis = {
            "question_type": self._classify_question_type(question),
            "key_entities": self._extract_entities(question),
            "required_tools": self._identify_required_tools(question),
            "complexity_factors": self._identify_complexity_factors(question)
        }

        return {
            "analysis": analysis,
            "next_action": "proceed_to_planning" if analysis["complexity_factors"] else "direct_execution"
        }

    def _plan_step(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Step de planejamento."""
        analysis = input_data.get("analysis", {})

        plan = {
            "approach": self._determine_approach(analysis),
            "tool_sequence": analysis.get("required_tools", []),
            "expected_challenges": self._identify_challenges(analysis),
            "success_criteria": self._define_success_criteria(analysis)
        }

        return {"plan": plan, "ready_for_execution": True}

    def _execute_step(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Step de execução principal."""
        question = input_data.get("question") or context.get("question", "")
        plan = input_data.get("plan", {})

        # Se há um plano, seguir a sequência de tools
        if plan and plan.get("tool_sequence"):
            results = []
            for tool_name in plan["tool_sequence"]:
                tool_result = self._execute_tool(tool_name, question, context)
                results.append({
                    "tool": tool_name,
                    "result": tool_result,
                    "success": tool_result.get("success", True)
                })

            return {
                "execution_results": results,
                "primary_result": results[0]["result"] if results else None
            }
        else:
            # Execução direta
            return self._direct_execution(question, context)

    def _reflect_step(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Step de reflexão sobre os resultados."""
        execution_results = input_data.get("execution_results", [])

        reflection = {
            "quality_assessment": self._assess_result_quality(execution_results),
            "completeness": self._assess_completeness(execution_results, context),
            "confidence_level": self._calculate_confidence(execution_results),
            "improvement_suggestions": self._suggest_improvements(execution_results)
        }

        return {
            "reflection": reflection,
            "needs_improvement": reflection["confidence_level"] < 0.7
        }

    def _decide_step(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Step de tomada de decisão."""
        analysis = input_data.get("analysis", {})

        decision = {
            "primary_strategy": self._choose_primary_strategy(analysis),
            "fallback_strategies": self._identify_fallbacks(analysis),
            "resource_requirements": self._estimate_resources(analysis),
            "risk_assessment": self._assess_risks(analysis)
        }

        return {"decision": decision}

    def _gather_info_step(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Step de coleta de informações."""
        question = input_data.get("question") or context.get("question", "")

        # Executar tools de busca de informação
        info_sources = []

        # Tentar busca na knowledge base se disponível
        if any(tool.name == "search_knowledge_base" for tool in self.tools):
            kb_result = self._execute_tool("search_knowledge_base", question, context)
            info_sources.append({
                "source": "knowledge_base",
                "result": kb_result,
                "relevance": self._calculate_relevance(kb_result, question)
            })

        return {
            "gathered_info": info_sources,
            "info_quality": self._assess_info_quality(info_sources)
        }

    def _synthesize_step(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Step de síntese final."""
        all_results = []

        # Coletar todos os resultados dos steps anteriores
        for key in ["execution_results", "gathered_info", "analysis", "plan"]:
            if key in input_data:
                all_results.append(input_data[key])

        # Gerar síntese usando LLM se disponível
        if self.llm_provider and all_results:
            synthesis_prompt = self._build_synthesis_prompt(context.get("question", ""), all_results)
            synthesis_result = self._call_llm(synthesis_prompt)
        else:
            # Síntese básica sem LLM
            synthesis_result = self._basic_synthesis(all_results)

        return {
            "synthesis": synthesis_result,
            "confidence": self._calculate_synthesis_confidence(all_results)
        }

    # Métodos auxiliares
    def _classify_question_type(self, question: str) -> str:
        """Classifica o tipo da pergunta."""
        question_lower = question.lower()

        if any(word in question_lower for word in ["o que é", "definição", "significado"]):
            return "definition"
        elif any(word in question_lower for word in ["como", "como fazer"]):
            return "how_to"
        elif any(word in question_lower for word in ["por que", "porque"]):
            return "explanation"
        elif any(word in question_lower for word in ["clima", "temperatura", "tempo"]):
            return "weather"
        else:
            return "general"

    def _extract_entities(self, question: str) -> List[str]:
        """Extrai entidades importantes da pergunta."""
        # Implementação simples - pode ser melhorada com NLP
        important_words = []
        skip_words = ["o", "que", "é", "como", "por", "que", "de", "da", "do", "na", "no", "em"]

        words = question.lower().split()
        for word in words:
            if len(word) > 3 and word not in skip_words:
                important_words.append(word)

        return important_words[:5]  # Top 5 entidades

    def _identify_required_tools(self, question: str) -> List[str]:
        """Identifica tools necessárias baseado na pergunta."""
        required_tools = []
        question_lower = question.lower()

        # Mapear tipos de pergunta para tools
        tool_mappings = {
            "search_knowledge_base": ["python", "programação", "langgraph", "conceito", "o que é"],
            "get_weather": ["clima", "tempo", "temperatura", "chuva"],
        }

        for tool_name, keywords in tool_mappings.items():
            if any(keyword in question_lower for keyword in keywords):
                required_tools.append(tool_name)

        # Se nenhuma tool específica foi identificada, usar conhecimento
        if not required_tools:
            required_tools = ["search_knowledge_base"]

        return required_tools

    def _identify_complexity_factors(self, question: str) -> List[str]:
        """Identifica fatores que aumentam a complexidade."""
        factors = []
        question_lower = question.lower()

        complexity_indicators = {
            "multiple_steps": ["passo a passo", "etapas", "processo"],
            "comparison": ["compare", "diferença", "melhor", "versus"],
            "integration": ["integre", "combine", "junto"],
            "analysis": ["analise", "avalie", "examine"]
        }

        for factor, indicators in complexity_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                factors.append(factor)

        return factors

    def _execute_tool(self, tool_name: str, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Executa uma tool específica."""
        try:
            # Encontrar a tool
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                return {"success": False, "error": f"Tool {tool_name} não encontrada"}

            # Executar tool
            if tool_name == "search_knowledge_base":
                result = tool.run(query=question)
            elif tool_name == "get_weather":
                # Extrair localização da pergunta
                location = self._extract_location(question)
                result = tool.run(location=location or "São Paulo")
            else:
                result = tool.run(input=question)

            return {"success": True, "result": result, "tool": tool_name}

        except Exception as e:
            return {"success": False, "error": str(e), "tool": tool_name}

    def _extract_location(self, question: str) -> Optional[str]:
        """Extrai localização de uma pergunta sobre clima."""
        # Implementação simples - pode ser melhorada
        words = question.split()
        prepositions = ["em", "de", "da", "do", "na", "no"]

        for i, word in enumerate(words):
            if word.lower() in prepositions and i + 1 < len(words):
                return words[i + 1]

        return None

    def _direct_execution(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execução direta sem planejamento complexo."""
        # Determinar melhor tool baseado na pergunta
        required_tools = self._identify_required_tools(question)

        if required_tools:
            return self._execute_tool(required_tools[0], question, context)
        else:
            return {"success": False, "error": "Nenhuma tool apropriada identificada"}

    def _call_llm(self, prompt: str) -> str:
        """Chama o LLM para geração de texto."""
        if self.llm_provider:
            try:
                return self.llm_provider.invoke(prompt)
            except Exception as e:
                return f"Erro ao chamar LLM: {e}"
        else:
            return "LLM não disponível"

    def _build_synthesis_prompt(self, question: str, results: List[Any]) -> str:
        """Constrói prompt para síntese final."""
        return f"""
Baseado na pergunta: "{question}"

E nos seguintes resultados de análise:
{json.dumps(results, indent=2, default=str)}

Forneça uma resposta clara, completa e bem estruturada em português brasileiro.
"""

    def _basic_synthesis(self, results: List[Any]) -> str:
        """Síntese básica sem LLM."""
        # Extrair resultado mais relevante
        for result in results:
            if isinstance(result, dict):
                if "result" in result:
                    return str(result["result"])
                elif "primary_result" in result:
                    return str(result["primary_result"])

        return "Informações processadas com sucesso"

    # Métodos de avaliação (implementações básicas)
    def _assess_result_quality(self, results: List[Dict]) -> float:
        success_count = sum(1 for r in results if r.get("success", True))
        return success_count / len(results) if results else 0.0

    def _assess_completeness(self, results: List[Dict], context: Dict) -> float:
        return 0.8  # Implementação placeholder

    def _calculate_confidence(self, results: List[Dict]) -> float:
        return self._assess_result_quality(results)

    def _suggest_improvements(self, results: List[Dict]) -> List[str]:
        return []  # Implementação placeholder

    def _choose_primary_strategy(self, analysis: Dict) -> str:
        return "tool_execution"  # Implementação placeholder

    def _identify_fallbacks(self, analysis: Dict) -> List[str]:
        return ["direct_response"]  # Implementação placeholder

    def _estimate_resources(self, analysis: Dict) -> Dict[str, Any]:
        return {"time": "low", "complexity": "medium"}

    def _assess_risks(self, analysis: Dict) -> Dict[str, Any]:
        return {"level": "low", "factors": []}

    def _calculate_relevance(self, result: Dict, question: str) -> float:
        return 0.8  # Implementação placeholder

    def _assess_info_quality(self, info_sources: List[Dict]) -> float:
        return 0.8  # Implementação placeholder

    def _calculate_synthesis_confidence(self, results: List[Any]) -> float:
        return 0.8  # Implementação placeholder


class AdvancedReasoningOrchestrator:
    """Orquestrador principal do sistema de reasoning avançado."""

    def __init__(self, executor: ReasoningExecutor = None):
        self.analyzer = ReasoningAnalyzer()
        self.executor = executor or DefaultReasoningExecutor()
        self.active_traces: Dict[str, ReasoningTrace] = {}

    @require_feature("advanced_reasoning", fallback_return="Reasoning avançado não disponível")
    def process_question(self, question: str, context: Dict[str, Any] = None,
                        trace_id: str = None) -> ReasoningTrace:
        """
        Processa uma pergunta usando reasoning avançado.

        Args:
            question: Pergunta a ser processada
            context: Contexto adicional
            trace_id: ID do trace (gerado se não fornecido)

        Returns:
            ReasoningTrace com o resultado completo
        """
        if ENHANCED_FEATURES_AVAILABLE:
            LogContext.set_operation_name("advanced_reasoning")

        context = context or {}
        trace_id = trace_id or f"reasoning_{int(time.time() * 1000)}"

        # Analisar complexidade
        complexity = self.analyzer.analyze_complexity(question)
        logger.info(f"Complexidade identificada: {complexity.value} para pergunta: {question[:50]}...")

        # Criar trace
        trace = ReasoningTrace(
            trace_id=trace_id,
            question=question,
            complexity=complexity,
            metadata={"context": context}
        )
        self.active_traces[trace_id] = trace

        try:
            # Obter steps sugeridos
            suggested_steps = self.analyzer.suggest_steps(question, complexity)
            logger.info(f"Steps planejados: {[s.value for s in suggested_steps]}")

            # Executar reasoning steps
            step_context = {"question": question, **context}

            for i, step_type in enumerate(suggested_steps):
                step_id = f"{trace_id}_step_{i+1}"

                # Criar step
                step = ReasoningStep(
                    step_id=step_id,
                    step_type=step_type,
                    description=self._get_step_description(step_type, question),
                    input_data=self._prepare_step_input(step_type, step_context, trace)
                )

                # Executar step
                start_time = time.time()
                try:
                    step.output_data = self.executor.execute_step(
                        step_type, step.input_data, step_context
                    )
                    step.execution_time = time.time() - start_time
                    step.success = True

                    # Atualizar contexto para próximo step
                    step_context.update(step.output_data)

                except Exception as e:
                    step.execution_time = time.time() - start_time
                    step.success = False
                    step.error_message = str(e)
                    logger.error(f"Erro no step {step_type.value}: {e}")

                trace.add_step(step)

                # Se step falhou e é crítico, parar
                if not step.success and step_type in [ReasoningStepType.EXECUTE]:
                    break

            # Completar trace
            final_answer = self._extract_final_answer(trace)
            confidence = self._calculate_final_confidence(trace)

            trace.complete(final_answer, confidence)

            return trace

        except Exception as e:
            trace.fail(str(e))
            logger.error(f"Reasoning falhou: {e}")
            return trace

        finally:
            # Limpeza
            if trace_id in self.active_traces:
                del self.active_traces[trace_id]

    def get_reasoning_trace(self, trace_id: str) -> Optional[ReasoningTrace]:
        """Obtém trace de reasoning por ID."""
        return self.active_traces.get(trace_id)

    def _get_step_description(self, step_type: ReasoningStepType, question: str) -> str:
        """Gera descrição para um step."""
        descriptions = {
            ReasoningStepType.ANALYZE: f"Analisando pergunta: {question[:50]}...",
            ReasoningStepType.PLAN: "Planejando abordagem de solução",
            ReasoningStepType.EXECUTE: "Executando solução planejada",
            ReasoningStepType.REFLECT: "Refletindo sobre resultados",
            ReasoningStepType.DECIDE: "Tomando decisão sobre estratégia",
            ReasoningStepType.GATHER_INFO: "Coletando informações relevantes",
            ReasoningStepType.SYNTHESIZE: "Sintetizando resposta final"
        }
        return descriptions.get(step_type, f"Executando step: {step_type.value}")

    def _prepare_step_input(self, step_type: ReasoningStepType,
                           context: Dict[str, Any], trace: ReasoningTrace) -> Dict[str, Any]:
        """Prepara input para um step baseado no contexto e steps anteriores."""
        input_data = {"question": context.get("question", "")}

        # Adicionar outputs dos steps anteriores relevantes
        for step in trace.steps:
            if step.success and step.output_data:
                input_data.update(step.output_data)

        return input_data

    def _extract_final_answer(self, trace: ReasoningTrace) -> str:
        """Extrai resposta final do trace."""
        # Procurar na ordem reversa pelo melhor resultado
        for step in reversed(trace.steps):
            if step.success and step.output_data:
                # Procurar campos que indicam resposta final
                for field in ["synthesis", "result", "primary_result", "execution_results"]:
                    if field in step.output_data:
                        result = step.output_data[field]
                        if isinstance(result, str):
                            return result
                        elif isinstance(result, dict) and "result" in result:
                            return str(result["result"])
                        elif isinstance(result, list) and result:
                            return str(result[0].get("result", result[0]))

        return "Não foi possível gerar uma resposta conclusiva."

    def _calculate_final_confidence(self, trace: ReasoningTrace) -> float:
        """Calcula confiança final baseada nos steps."""
        if not trace.steps:
            return 0.0

        successful_steps = [s for s in trace.steps if s.success]
        if not successful_steps:
            return 0.0

        # Média de confiança dos steps bem-sucedidos
        total_confidence = sum(step.confidence for step in successful_steps)
        return total_confidence / len(successful_steps)


# Instância global para uso fácil
_global_orchestrator = None

def get_reasoning_orchestrator(llm_provider=None, tools=None) -> AdvancedReasoningOrchestrator:
    """Obtém instância global do orquestrador de reasoning."""
    global _global_orchestrator
    if _global_orchestrator is None:
        executor = DefaultReasoningExecutor(llm_provider, tools)
        _global_orchestrator = AdvancedReasoningOrchestrator(executor)
    return _global_orchestrator


# Função de conveniência
def reason_about(question: str, context: Dict[str, Any] = None,
                llm_provider=None, tools=None) -> ReasoningTrace:
    """Função de conveniência para reasoning avançado."""
    orchestrator = get_reasoning_orchestrator(llm_provider, tools)
    return orchestrator.process_question(question, context)


if __name__ == "__main__":
    # Exemplo de uso do sistema de reasoning avançado
    print("🧠 Testando Sistema de Reasoning Avançado")

    # Criar orquestrador
    orchestrator = AdvancedReasoningOrchestrator()

    # Testar diferentes níveis de complexidade
    test_questions = [
        "O que é Python?",  # Simple
        "Como implementar uma API REST?",  # Moderate
        "Analise e compare diferentes arquiteturas de microserviços",  # Complex
        "Desenvolva uma estratégia completa de migração para cloud"  # Advanced
    ]

    for question in test_questions:
        print(f"\n📝 Pergunta: {question}")

        try:
            trace = orchestrator.process_question(question)

            print(f"🎯 Complexidade: {trace.complexity.value}")
            print(f"📊 Steps executados: {len(trace.steps)}")
            print(f"⏱️ Tempo total: {trace.total_time:.2f}s")
            print(f"🎪 Confiança: {trace.confidence:.1%}")

            if trace.success:
                print(f"✅ Resposta: {trace.final_answer[:100]}...")
            else:
                print(f"❌ Erro: {trace.error_message}")

        except Exception as e:
            print(f"❌ Erro no teste: {e}")

    print("\n🎯 Sistema de Reasoning Avançado configurado e testado!")