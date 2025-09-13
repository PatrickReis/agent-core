#!/usr/bin/env python3
"""
Eval Reporter - Gerador de relatórios para resultados de avaliações.
Cria relatórios em múltiplos formatos (JSON, HTML, CSV, Markdown).

Usage:
    python utils/eval_reporter.py --input results.json --format html
    python utils/eval_reporter.py --directory eval_results --format all
    python utils/eval_reporter.py --comparison comparison_results.json --output report.html
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics

# Adicionar o diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent))


class EvalReporter:
    """Gerador de relatórios para resultados de avaliações."""

    def __init__(self):
        self.templates_dir = Path(__file__).parent / "templates"
        self.templates_dir.mkdir(exist_ok=True)
        self._create_html_template()

    def generate_single_eval_report(self, result_data: Dict[str, Any],
                                   format_type: str = "html",
                                   output_file: Optional[str] = None) -> str:
        """Gera relatório para uma avaliação individual."""

        if format_type == "html":
            return self._generate_html_single_report(result_data, output_file)
        elif format_type == "json":
            return self._generate_json_report(result_data, output_file)
        elif format_type == "csv":
            return self._generate_csv_single_report(result_data, output_file)
        elif format_type == "markdown":
            return self._generate_markdown_single_report(result_data, output_file)
        else:
            raise ValueError(f"Formato não suportado: {format_type}")

    def generate_comparison_report(self, comparison_data: Dict[str, Any],
                                 format_type: str = "html",
                                 output_file: Optional[str] = None) -> str:
        """Gera relatório para comparação entre modelos."""

        if format_type == "html":
            return self._generate_html_comparison_report(comparison_data, output_file)
        elif format_type == "json":
            return self._generate_json_report(comparison_data, output_file)
        elif format_type == "csv":
            return self._generate_csv_comparison_report(comparison_data, output_file)
        elif format_type == "markdown":
            return self._generate_markdown_comparison_report(comparison_data, output_file)
        else:
            raise ValueError(f"Formato não suportado: {format_type}")

    def _generate_html_single_report(self, result_data: Dict[str, Any],
                                   output_file: Optional[str] = None) -> str:
        """Gera relatório HTML para avaliação individual."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"eval_report_{timestamp}.html"

        # Extrair dados principais
        summary = result_data.get('summary', {})
        metrics = summary.get('metrics', {})
        overall_score = summary.get('overall_score', 0)
        overall_grade = summary.get('overall_grade', 'N/A')
        execution_time = summary.get('execution_time', 0)

        # Template HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Avaliação - {summary.get('suite', 'N/A')}</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Relatório de Avaliação</h1>
            <div class="header-info">
                <span>🤖 Agente: {summary.get('agent', 'N/A')}</span>
                <span>📝 Suite: {summary.get('suite', 'N/A')}</span>
                <span>📅 Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}</span>
            </div>
        </header>

        <section class="overview">
            <h2>📈 Visão Geral</h2>
            <div class="score-card">
                <div class="score-main">
                    <div class="score-value grade-{overall_grade.lower()}">{overall_score:.1f}%</div>
                    <div class="score-grade">Nota {overall_grade}</div>
                </div>
                <div class="score-details">
                    <div class="detail-item">
                        <span class="label">Total de Testes:</span>
                        <span class="value">{summary.get('total_tests', 0)}</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Tempo de Execução:</span>
                        <span class="value">{execution_time:.2f}s</span>
                    </div>
                </div>
            </div>
        </section>

        <section class="metrics">
            <h2>🎯 Métricas Detalhadas</h2>
            <div class="metrics-grid">
                {self._generate_metrics_html(metrics)}
            </div>
        </section>

        <section class="analysis">
            <h2>🔍 Análise</h2>
            {self._generate_analysis_html(metrics, overall_score)}
        </section>

        <footer>
            <p>Relatório gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')} pelo Agent Core Eval System</p>
        </footer>
    </div>
</body>
</html>"""

        # Salvar arquivo
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📄 Relatório HTML gerado: {output_file}")
        return output_file

    def _generate_html_comparison_report(self, comparison_data: Dict[str, Any],
                                       output_file: Optional[str] = None) -> str:
        """Gera relatório HTML para comparação entre modelos."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"comparison_report_{timestamp}.html"

        # Extrair dados
        models = comparison_data.get('models', [])
        suite_name = comparison_data.get('suite_name', 'N/A')
        winner = comparison_data.get('winner', 'N/A')
        individual_results = comparison_data.get('individual_results', {})
        comparison_metrics = comparison_data.get('comparison_metrics', {})

        # Preparar dados para visualização
        model_scores = []
        for model in models:
            if model in individual_results:
                result = individual_results[model]
                if 'summary' in result:
                    model_scores.append({
                        'model': model,
                        'score': result['summary'].get('overall_score', 0),
                        'grade': result['summary'].get('overall_grade', 'F'),
                        'time': result['summary'].get('execution_time', 0)
                    })

        model_scores.sort(key=lambda x: x['score'], reverse=True)

        html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Comparação - {suite_name}</title>
    <style>
        {self._get_css_styles()}
        {self._get_comparison_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏆 Relatório de Comparação de Modelos</h1>
            <div class="header-info">
                <span>📝 Suite: {suite_name}</span>
                <span>🤖 Modelos: {len(models)}</span>
                <span>📅 Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}</span>
            </div>
        </header>

        <section class="winner">
            <h2>🥇 Vencedor</h2>
            <div class="winner-card">
                <div class="winner-name">{winner}</div>
                <div class="winner-score">{next((m['score'] for m in model_scores if m['model'] == winner), 0):.1f}%</div>
            </div>
        </section>

        <section class="ranking">
            <h2>📊 Ranking dos Modelos</h2>
            <div class="ranking-table">
                {self._generate_ranking_html(model_scores)}
            </div>
        </section>

        <section class="metrics-comparison">
            <h2>📈 Comparação por Métrica</h2>
            {self._generate_metrics_comparison_html(comparison_metrics.get('by_metric', {}))}
        </section>

        <section class="statistics">
            <h2>📋 Estatísticas</h2>
            {self._generate_statistics_html(comparison_metrics)}
        </section>

        <footer>
            <p>Relatório gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')} pelo Agent Core Eval System</p>
        </footer>
    </div>
</body>
</html>"""

        # Salvar arquivo
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📄 Relatório de comparação HTML gerado: {output_file}")
        return output_file

    def _generate_csv_single_report(self, result_data: Dict[str, Any],
                                  output_file: Optional[str] = None) -> str:
        """Gera relatório CSV para avaliação individual."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"eval_report_{timestamp}.csv"

        summary = result_data.get('summary', {})
        metrics = summary.get('metrics', {})

        # Preparar dados para CSV
        rows = []

        # Linha de resumo
        rows.append([
            'RESUMO',
            summary.get('agent', 'N/A'),
            summary.get('suite', 'N/A'),
            summary.get('overall_score', 0),
            summary.get('overall_grade', 'N/A'),
            summary.get('total_tests', 0),
            summary.get('execution_time', 0)
        ])

        # Linhas de métricas
        for metric_name, metric_data in metrics.items():
            rows.append([
                'METRICA',
                metric_name,
                '',
                metric_data.get('mean', 0),
                '',
                metric_data.get('passed_count', 0),
                metric_data.get('pass_rate', 0)
            ])

        # Escrever CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Tipo', 'Nome/Agente', 'Suite/Métrica', 'Score', 'Grade', 'Testes/Aprovados', 'Tempo/Taxa'])
            writer.writerows(rows)

        print(f"📊 Relatório CSV gerado: {output_file}")
        return output_file

    def _generate_markdown_single_report(self, result_data: Dict[str, Any],
                                       output_file: Optional[str] = None) -> str:
        """Gera relatório Markdown para avaliação individual."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"eval_report_{timestamp}.md"

        summary = result_data.get('summary', {})
        metrics = summary.get('metrics', {})

        markdown_content = f"""# 📊 Relatório de Avaliação

**🤖 Agente:** {summary.get('agent', 'N/A')}
**📝 Suite:** {summary.get('suite', 'N/A')}
**📅 Data:** {datetime.now().strftime('%d/%m/%Y %H:%M')}

## 📈 Visão Geral

- **Score Geral:** {summary.get('overall_score', 0):.1f}% (Nota {summary.get('overall_grade', 'N/A')})
- **Total de Testes:** {summary.get('total_tests', 0)}
- **Tempo de Execução:** {summary.get('execution_time', 0):.2f}s

## 🎯 Métricas Detalhadas

| Métrica | Score Médio | Taxa de Aprovação | Min | Max |
|---------|-------------|------------------|-----|-----|
{self._generate_metrics_table(metrics)}

## 🔍 Análise

{self._generate_markdown_analysis(metrics, summary.get('overall_score', 0))}

---
*Relatório gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')} pelo Agent Core Eval System*
"""

        # Salvar arquivo
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        print(f"📝 Relatório Markdown gerado: {output_file}")
        return output_file

    def _generate_json_report(self, data: Dict[str, Any],
                            output_file: Optional[str] = None) -> str:
        """Gera relatório JSON (estruturado)."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"eval_report_{timestamp}.json"

        # Adicionar metadados do relatório
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "Agent Core Eval System",
                "version": "1.0.0"
            },
            "data": data
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"📄 Relatório JSON gerado: {output_file}")
        return output_file

    def _get_css_styles(self) -> str:
        """Retorna estilos CSS para os relatórios HTML."""
        return """
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f7fa;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }

        header h1 {
            font-size: 2.5em;
            margin-bottom: 15px;
        }

        .header-info {
            display: flex;
            justify-content: center;
            gap: 30px;
            font-size: 1.1em;
        }

        section {
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        h2 {
            color: #2d3748;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
            padding-left: 15px;
        }

        .score-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 30px;
            border-radius: 15px;
            color: white;
        }

        .score-value {
            font-size: 3.5em;
            font-weight: bold;
        }

        .score-grade {
            font-size: 1.2em;
            margin-top: 5px;
        }

        .grade-a { color: #48bb78; }
        .grade-b { color: #38b2ac; }
        .grade-c { color: #ed8936; }
        .grade-d { color: #e53e3e; }
        .grade-f { color: #9f1239; }

        .detail-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 1.1em;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }

        .metric-card {
            border: 1px solid #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            background: #f8fafc;
        }

        .metric-name {
            font-weight: bold;
            font-size: 1.2em;
            margin-bottom: 10px;
            color: #2d3748;
        }

        .metric-score {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }

        .pass { color: #48bb78; }
        .warn { color: #ed8936; }
        .fail { color: #e53e3e; }

        .progress-bar {
            background: #e2e8f0;
            height: 10px;
            border-radius: 5px;
            overflow: hidden;
            margin: 10px 0;
        }

        .progress-fill {
            height: 100%;
            border-radius: 5px;
            transition: width 0.3s ease;
        }

        footer {
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #e2e8f0;
        }
        """

    def _get_comparison_css(self) -> str:
        """CSS adicional para relatórios de comparação."""
        return """
        .winner-card {
            text-align: center;
            background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
            padding: 40px;
            border-radius: 15px;
            color: #2d3748;
        }

        .winner-name {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .winner-score {
            font-size: 2em;
            font-weight: bold;
        }

        .ranking-table {
            overflow-x: auto;
        }

        .ranking-item {
            display: flex;
            align-items: center;
            padding: 15px;
            margin: 10px 0;
            background: #f8fafc;
            border-radius: 8px;
            border-left: 5px solid;
        }

        .rank-1 { border-left-color: #ffd700; }
        .rank-2 { border-left-color: #c0c0c0; }
        .rank-3 { border-left-color: #cd7f32; }
        .rank-other { border-left-color: #94a3b8; }

        .rank-position {
            font-size: 1.5em;
            font-weight: bold;
            margin-right: 20px;
            min-width: 50px;
        }

        .model-info {
            flex-grow: 1;
        }

        .model-name {
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .model-stats {
            color: #666;
        }
        """

    def _generate_metrics_html(self, metrics: Dict[str, Any]) -> str:
        """Gera HTML para exibição de métricas."""
        html_parts = []

        for metric_name, metric_data in metrics.items():
            mean_score = metric_data.get('mean', 0)
            pass_rate = metric_data.get('pass_rate', 0)

            # Determinar classe CSS baseada no score
            if mean_score >= 80:
                score_class = "pass"
                progress_class = "pass"
            elif mean_score >= 60:
                score_class = "warn"
                progress_class = "warn"
            else:
                score_class = "fail"
                progress_class = "fail"

            html_parts.append(f"""
            <div class="metric-card">
                <div class="metric-name">{metric_name.title()}</div>
                <div class="metric-score {score_class}">{mean_score:.1f}%</div>
                <div class="progress-bar">
                    <div class="progress-fill {progress_class}" style="width: {mean_score}%"></div>
                </div>
                <div class="detail-item">
                    <span>Taxa de Aprovação:</span>
                    <span>{pass_rate:.1f}%</span>
                </div>
                <div class="detail-item">
                    <span>Min/Max:</span>
                    <span>{metric_data.get('min', 0):.1f}% / {metric_data.get('max', 0):.1f}%</span>
                </div>
            </div>
            """)

        return "\n".join(html_parts)

    def _generate_analysis_html(self, metrics: Dict[str, Any], overall_score: float) -> str:
        """Gera análise em HTML."""
        analysis_parts = []

        # Análise geral
        if overall_score >= 85:
            analysis_parts.append("<div class='alert success'>🎉 <strong>Excelente performance!</strong> O agente demonstrou alta qualidade em todas as métricas.</div>")
        elif overall_score >= 70:
            analysis_parts.append("<div class='alert info'>👍 <strong>Boa performance.</strong> Resultados satisfatórios com algumas áreas para melhoria.</div>")
        else:
            analysis_parts.append("<div class='alert warning'>⚠️ <strong>Performance abaixo do esperado.</strong> Várias áreas precisam de atenção.</div>")

        # Análise por métrica
        for metric_name, metric_data in metrics.items():
            mean_score = metric_data.get('mean', 0)
            if mean_score < 70:
                analysis_parts.append(f"<div class='alert warning'>🔍 <strong>{metric_name.title()}:</strong> Score baixo ({mean_score:.1f}%) - necessita melhoria.</div>")

        return "\n".join(analysis_parts) if analysis_parts else "<p>Análise detalhada não disponível.</p>"

    def _generate_ranking_html(self, model_scores: List[Dict[str, Any]]) -> str:
        """Gera HTML para ranking de modelos."""
        html_parts = []

        for i, model_data in enumerate(model_scores, 1):
            rank_class = f"rank-{i}" if i <= 3 else "rank-other"
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}°"

            html_parts.append(f"""
            <div class="ranking-item {rank_class}">
                <div class="rank-position">{medal}</div>
                <div class="model-info">
                    <div class="model-name">{model_data['model']}</div>
                    <div class="model-stats">
                        Score: {model_data['score']:.1f}% |
                        Nota: {model_data['grade']} |
                        Tempo: {model_data['time']:.2f}s
                    </div>
                </div>
            </div>
            """)

        return "\n".join(html_parts)

    def _generate_metrics_comparison_html(self, metrics_comparison: Dict[str, Dict[str, float]]) -> str:
        """Gera HTML para comparação de métricas."""
        if not metrics_comparison:
            return "<p>Dados de comparação não disponíveis.</p>"

        html_parts = []

        for metric_name, model_scores in metrics_comparison.items():
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

            html_parts.append(f"""
            <div class="metric-comparison">
                <h3>{metric_name.title()}</h3>
                <div class="comparison-bars">
            """)

            max_score = max(model_scores.values()) if model_scores.values() else 1

            for model, score in sorted_models:
                bar_width = (score / max_score) * 100 if max_score > 0 else 0
                html_parts.append(f"""
                    <div class="comparison-bar">
                        <span class="model-label">{model}</span>
                        <div class="bar-container">
                            <div class="bar-fill" style="width: {bar_width}%"></div>
                            <span class="score-label">{score:.1f}%</span>
                        </div>
                    </div>
                """)

            html_parts.append("</div></div>")

        return "\n".join(html_parts)

    def _generate_statistics_html(self, comparison_metrics: Dict[str, Any]) -> str:
        """Gera HTML para estatísticas de comparação."""
        if not comparison_metrics:
            return "<p>Estatísticas não disponíveis.</p>"

        return f"""
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{comparison_metrics.get('best_score', 0):.1f}%</div>
                <div class="stat-label">Melhor Score</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{comparison_metrics.get('worst_score', 0):.1f}%</div>
                <div class="stat-label">Pior Score</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{comparison_metrics.get('average_score', 0):.1f}%</div>
                <div class="stat-label">Score Médio</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{comparison_metrics.get('score_range', 0):.1f}%</div>
                <div class="stat-label">Diferença</div>
            </div>
        </div>
        """

    def _generate_metrics_table(self, metrics: Dict[str, Any]) -> str:
        """Gera tabela Markdown para métricas."""
        rows = []
        for metric_name, metric_data in metrics.items():
            rows.append(f"| {metric_name.title()} | {metric_data.get('mean', 0):.1f}% | {metric_data.get('pass_rate', 0):.1f}% | {metric_data.get('min', 0):.1f}% | {metric_data.get('max', 0):.1f}% |")

        return "\n".join(rows) if rows else "| Nenhuma métrica disponível | - | - | - | - |"

    def _generate_markdown_analysis(self, metrics: Dict[str, Any], overall_score: float) -> str:
        """Gera análise em Markdown."""
        analysis_parts = []

        if overall_score >= 85:
            analysis_parts.append("### ✅ Análise Positiva\n- Excelente performance geral")
        elif overall_score >= 70:
            analysis_parts.append("### ⚠️ Análise Moderada\n- Performance satisfatória com potencial de melhoria")
        else:
            analysis_parts.append("### ❌ Áreas de Melhoria\n- Performance abaixo do esperado")

        # Identificar métricas problemáticas
        problem_metrics = [name for name, data in metrics.items() if data.get('mean', 0) < 70]
        if problem_metrics:
            analysis_parts.append(f"\n### 🔍 Métricas que precisam de atenção:\n" +
                                 "\n".join([f"- **{metric}**: Requer melhoria" for metric in problem_metrics]))

        return "\n".join(analysis_parts) if analysis_parts else "Análise não disponível."

    def _create_html_template(self):
        """Cria templates HTML básicos."""
        # Template será criado dinamicamente nos métodos de geração


def main():
    """Interface principal de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Gerador de Relatórios para Avaliações",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python utils/eval_reporter.py --input results.json --format html
  python utils/eval_reporter.py --input comparison.json --format all
  python utils/eval_reporter.py --directory eval_results --format html
        """
    )

    parser.add_argument("--input", help="Arquivo JSON com resultados")
    parser.add_argument("--directory", help="Diretório com múltiplos arquivos de resultado")
    parser.add_argument("--format", choices=["html", "json", "csv", "markdown", "all"],
                       default="html", help="Formato do relatório")
    parser.add_argument("--output", help="Nome do arquivo de saída")
    parser.add_argument("--comparison", action="store_true",
                       help="Tratar como dados de comparação")

    args = parser.parse_args()

    if not args.input and not args.directory:
        parser.print_help()
        print("\n❌ Especifique --input ou --directory")
        return 1

    reporter = EvalReporter()

    try:
        if args.input:
            # Processar arquivo único
            with open(args.input, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if args.format == "all":
                formats = ["html", "json", "csv", "markdown"]
            else:
                formats = [args.format]

            for format_type in formats:
                if args.comparison:
                    reporter.generate_comparison_report(data, format_type, args.output)
                else:
                    reporter.generate_single_eval_report(data, format_type, args.output)

        elif args.directory:
            # Processar diretório
            directory = Path(args.directory)
            if not directory.exists():
                print(f"❌ Diretório não encontrado: {args.directory}")
                return 1

            json_files = list(directory.glob("*.json"))
            if not json_files:
                print(f"❌ Nenhum arquivo JSON encontrado em: {args.directory}")
                return 1

            print(f"📁 Processando {len(json_files)} arquivos...")

            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                output_name = f"report_{json_file.stem}.html"
                if args.comparison:
                    reporter.generate_comparison_report(data, args.format, output_name)
                else:
                    reporter.generate_single_eval_report(data, args.format, output_name)

        print("✅ Relatórios gerados com sucesso!")
        return 0

    except Exception as e:
        print(f"❌ Erro ao gerar relatórios: {e}")
        return 1


if __name__ == "__main__":
    exit(main())