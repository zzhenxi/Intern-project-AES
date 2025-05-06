from evaluation_system import evaluate_essays
from data_utils import load_dataset
from utils import analyze_results, save_results
from agents import AgentConfig
import logging


logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("essay_evaluation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    api_key = ''
    dataset_name = 'asap'
    dataset_path = 'datasets/asap_essay_set_1.jsonl'
    n_samples = 1
    model_name = "gpt-4.1-mini"
    n_agents = 4
    feedback = True
    results_dir = 'results'
    
    dataset = load_dataset(dataset_name, dataset_path, n_samples)
    logger.info(f"Processing essay: {dataset_path}")

    if dataset_name == 'asap':
        sample = dataset[0]
        max_score = sample['max_score']
        min_score = sample['min_score']
        source_essay = True if sample['source_text'] else False
        grade_level = sample['grade_level']
        essay_type = sample['type_of_essay']

        essay_set_id = dataset_path.split('_')[-1].split('.')[0]
        dataset_name = dataset_name+'_'+essay_set_id ### dataset_name converted!

    elif dataset_name == '':
        pass
    
    config = AgentConfig(
        n_agents=n_agents,
        max_score=max_score,
        min_score=min_score,
        max_holistic_score=max_score,
        min_holistic_score=min_score,
        feedback=feedback,
        model_name=model_name,
        grade_level=grade_level,
        essay_type=essay_type
    )
    
    evaluation_results = evaluate_essays(dataset, config, api_key, source_essay)
    stats = analyze_results(evaluation_results)
    save_results(
        evaluation_results, 
        stats, 
        dataset_name,
        results_dir
    )
    
    logger.info(f"Completed processing essay: {dataset_path}")

if __name__ == "__main__":
    main()