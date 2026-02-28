#!/usr/bin/env python3
"""
CLI entry point for the Microservice Dependency Analyzer.

Usage:
  python run.py /path/to/project
  python run.py /path/to/project --no-llm
  python run.py /path/to/project --non-interactive
  python run.py /path/to/project --ground-truth data/ground_truth/gt.json
  python run.py /path/to/project --output data/outputs --graph data/graphs/graph.html
"""

import argparse
import sys
import os
from pathlib import Path

# Must be run from project root: python run.py ...
# OR: python -m run (from project root)
_root = str(Path(__file__).parent.resolve())
if _root not in sys.path:
    sys.path.insert(0, _root)

from agent.core import DependencyAnalyzer
from graph.output_writer import OutputWriter

try:
    from graph.builder import GraphBuilder
    from graph.visualizer import Visualizer
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False


def main():
    parser = argparse.ArgumentParser(
        description='Universal Microservice Dependency Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py ~/projects/my-microservices
  python run.py ~/projects/my-app --no-llm --non-interactive
  python run.py ~/projects/my-app --ground-truth data/ground_truth/gt.json
  LLM_BACKEND=gemini python run.py ~/projects/my-app
  LLM_BACKEND=ollama python run.py ~/projects/my-app --no-llm=false
        """
    )

    parser.add_argument('project_path', help='Path to the microservice project')
    parser.add_argument('--no-llm', action='store_true', default=False,
                        help='Skip LLM analysis (faster, offline)')
    parser.add_argument('--non-interactive', action='store_true', default=False,
                        help='Skip user questions (use defaults)')
    parser.add_argument('--output', default='data/outputs',
                        help='Output directory for reports')
    parser.add_argument('--graph', default=None,
                        help='Path for HTML graph visualization')
    parser.add_argument('--ground-truth', default=None,
                        help='Path to ground_truth.json for evaluation')
    parser.add_argument('--save-graph', default=None,
                        help='Save NetworkX graph to file')

    args = parser.parse_args()

    # Validate project path
    project_path = Path(args.project_path).resolve()
    if not project_path.exists():
        print(f"❌ Project path does not exist: {project_path}")
        sys.exit(1)
    if not project_path.is_dir():
        print(f"❌ Project path is not a directory: {project_path}")
        sys.exit(1)

    # Run analysis
    analyzer = DependencyAnalyzer(
        project_path=str(project_path),
        non_interactive=args.non_interactive,
        skip_llm=args.no_llm,
    )
    result = analyzer.analyze()

    if not result.services:
        print("❌ No services discovered. Exiting.")
        sys.exit(1)

    # Save reports
    print("\n💾 Saving outputs...")
    writer = OutputWriter(result, args.output)
    paths = writer.save_all()

    # Also save a fixed-name "latest" JSON so downstream tools (feature_engineering.py)
    # can find this project's output without needing the timestamp.
    import shutil
    project_name = project_path.name
    latest_json = Path(args.output) / f'{project_name}_latest.json'
    timed_json = paths.get('json', '')
    if timed_json and Path(timed_json).exists():
        shutil.copy2(timed_json, latest_json)
        print(f"   🔗 Latest JSON: {latest_json}")

    # Save graph visualization
    if HAS_GRAPH:
        graph_path = args.graph or str(Path(args.output) / 'graph.html')
        vis = Visualizer(result)
        vis.save_html(graph_path)

        if args.save_graph:
            try:
                import networkx as nx
                import pickle
                builder = GraphBuilder(result)
                G = builder.build()
                if G:
                    with open(args.save_graph, 'wb') as f:
                        pickle.dump(G, f)
                    print(f"   📦 Graph pickled: {args.save_graph}")
            except Exception as e:
                print(f"   ⚠️  Could not save graph: {e}")

    # Evaluate if ground truth provided
    if args.ground_truth:
        try:
            from evaluation.evaluator import Evaluator
            evaluator = Evaluator(result, args.ground_truth)
            eval_results = evaluator.evaluate()

            # Save eval results
            import json
            from datetime import datetime
            eval_path = Path(args.output) / f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            eval_path.write_text(json.dumps(eval_results, indent=2))
            print(f"   📏 Eval saved: {eval_path}")
        except Exception as e:
            print(f"   ⚠️  Evaluation failed: {e}")

    print("\n✅ Done!")
    print(f"   JSON: {paths.get('json', '?')}")
    print(f"   CSV:  {paths.get('csv', '?')}")
    print(f"   HTML: {paths.get('html', '?')}")


if __name__ == '__main__':
    main()
