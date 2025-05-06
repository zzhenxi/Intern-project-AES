import argparse
import logging
import os
from datetime import datetime


from evaluation_system import evaluate_essays
from data_utils import load_dataset
from utils import analyze_results, save_results
from agents import AgentConfig

# 로깅 설정
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("essay_evaluation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main(args):
    dataset = load_dataset(args.dataset_name, args.dataset_path, args.n_samples)
    logger.info(f"Processing essay: {args.dataset_path}")

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # args.results_dir = os.path.join(args.results_dir, str(timestamp))
    # if not os.path.exists(args.results_dir):
    #     os.makedirs(args.results_dir)

    if args.dataset_name == 'asap':
        sample = dataset[0]
        max_score = sample['max_score']
        min_score = sample['min_score']
        source_essay = True if sample.get('source_text') else False
        grade_level = 'grade '+str(sample['grade_level']).strip()
        essay_type = sample['type_of_essay']

        essay_set_id = os.path.splitext(os.path.basename(args.dataset_path))[0].split('_')[-1]
        dataset_name = f"{args.dataset_name}_{essay_set_id}"
    else:
        logger.error("Unsupported dataset")
        return

    config = AgentConfig(
        n_agents=args.n_agents,
        max_score=max_score,
        min_score=min_score,
        max_holistic_score=max_score,
        min_holistic_score=min_score,
        feedback=args.feedback,
        model_name=args.model_name,
        grade_level=grade_level,
        essay_type=essay_type,
        api_type=args.api_type,
        local_model_path=args.local_model_path if args.api_type == 'huggingface' else None
    )

    if args.api_type == 'openai':
        api_key = args.openai_api_key
    elif args.api_type == 'anthropic':
        api_key = args.claude_api_key
    elif args.api_type == 'huggingface':
        api_key = None  # No API key needed for local models
    else:
        logger.error("Unsupported API type")
        return

    evaluation_results = evaluate_essays(dataset, config, api_key, source_essay, args.multi_agent)
    stats = analyze_results(evaluation_results, args.multi_agent)
    save_results(evaluation_results, stats, dataset_name, args.results_dir, args.multi_agent)
    logger.info(f"Completed processing essay: {args.dataset_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Essay Evaluation System")
    parser.add_argument('--openai_api_key', type=str, default='', help='OpenAI API key')
    parser.add_argument('--claude_api_key', type=str, default='', help='Claude API key')
    parser.add_argument('--api_type', type=str, choices=['openai', 'anthropic', 'huggingface'], default='openai', help='API type to use')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g., "asap")')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset JSONL file')
    parser.add_argument('--n_samples', type=int, default=1, help='Number of samples to evaluate')
    parser.add_argument('--model_name', type=str, default='gpt-4.1-mini', help='Model to use for evaluation')
    parser.add_argument('--n_agents', type=int, default=4, help='Number of evaluator agents')
    parser.add_argument('--feedback', action='store_true', help='Whether to generate feedback')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--multi_agent', action='store_true', help='Use multi-agent evaluation')

    args = parser.parse_args()
    main(args)
