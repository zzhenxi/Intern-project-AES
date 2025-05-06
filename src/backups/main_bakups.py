from config import EssayEvaluationConfig
from agents import EssayEvaluationSystem
from datasets import utils
import json








def main():
    # Configuration
    
    api_key = ''
    dataset_name = 'asap'
    dataset_path = '/home/jinhee/NC/Intern-project-AES/datasets/preprocessed_essays.jsonl'
    n_samples = 10

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f][:n_samples]

    if dataset_name == 'asap':
        splited_dataset = {}
        for sample in dataset:
            essay_id = sample['essay_set']
            splited_key = 'essay_set_'+str(essay_id)
            if splited_key not in splited_dataset.keys():
                splited_dataset[splited_key] = []
            splited_dataset[splited_key].append(sample)
        
        for essay_set in splited_dataset.keys():
            max_score = splited_dataset[essay_set][0]['max_score'],
            min_score = splited_dataset[essay_set][0]['min_score']
            config = EssayEvaluationConfig(
                n_agents=4,
                max_score=max_score,
                min_score=min_score,
                max_holistic_score=max_score,
                min_holistic_score=min_score,
                feedback=True,
                model_name="gpt-4.1-mini"
            )

            system = EssayEvaluationSystem(
                api_key=api_key,  # Replace with your API key
                config=config
            )

            evaluation_results = []

            for data in splited_dataset[essay_set]:
                essay = data['essay']
                essay_prompt = data['prompt_text']
                evaluation_result = system.evaluate_essay(essay, essay_prompt)
                evaluation_results.append(json.dumps(evaluation_result, indent=2))
            





# Example usage
if __name__ == "__main__":
    main()
    # Configuration