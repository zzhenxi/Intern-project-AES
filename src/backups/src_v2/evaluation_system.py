from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
import logging
import os
from agents import AgentConfig, PersonaAgent, RubricAgent, ScoringAgent, MetaScoreAgent, SingleEvalAgent



logger = logging.getLogger(__name__)



class MultiAgentEssayEvaluationSystem:
    """Main system that orchestrates the multi-agent evaluation process"""
    
    def __init__(self, api_key: str = None, config: Optional[AgentConfig] = None):
        if api_key:
            if config.api_type == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = api_key
            else: 
                os.environ["OPENAI_API_KEY"] = api_key
            
        self.config = config or AgentConfig()
        self.persona_agent = PersonaAgent(self.config)
        self.rubric_agent = RubricAgent(self.config)
        self.scoring_agent = ScoringAgent(self.config)
        self.meta_agent = MetaScoreAgent(self.config)

    def evaluate_essay(self, essay: str, essay_prompt: str, source_essay: str = False) -> Optional[Dict[str, Any]]:
        """Full evaluation pipeline that returns structured assessment of an essay"""
        
        # Step 1: Generate specialized personas
        personas = self.persona_agent.generate_personas(essay, essay_prompt, source_essay)
        if not personas:
            logging.warning("Skipping: Persona generation failed.")
            return None

        # Step 2: Generate rubrics for each persona
        rubrics = self.rubric_agent.generate_rubrics(personas, essay, essay_prompt, source_essay)
        if not rubrics:
            logging.warning("Skipping: Rubric generation failed.")
            return None

        # Step 3: Score essay using each persona's rubric
        all_scores = self.scoring_agent.generate_scores(rubrics, essay, essay_prompt, source_essay)
        if not all_scores:
            logging.warning("Skipping: Scoring failed.")
            return None

        # Step 4: Generate meta-assessment with final score
        final_assessment = self.meta_agent.generate_meta_score(all_scores, essay, essay_prompt, source_essay)
        if not final_assessment:
            logging.warning("Skipping: Meta-assessment failed.")
            return None

        # Compile complete evaluation result
        evaluation_result = {
            "personas": personas,
            "rubrics": rubrics,
            "detailed_scores": all_scores,
            "final_assessment": final_assessment,
            "has_source_essay": source_essay is not None
        }

        return evaluation_result


class SingleAgentEssayEvaluationSystem:
    """Simplified evaluation system that uses a single agent for essay evaluation"""
    
    def __init__(self, api_key: str = None, config: Optional[AgentConfig] = None):
        if api_key:
            if config.api_type == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = api_key
            else: 
                os.environ["OPENAI_API_KEY"] = api_key
            
        self.config = config or AgentConfig()
        self.evaluator = SingleEvalAgent(self.config)
    
    def evaluate_essay(self, essay: str, essay_prompt: str, source_essay: str = False) -> Optional[Dict[str, Any]]:
        final_assessment = self.evaluator.generate_holistic_score(essay, essay_prompt, source_essay)
        if not final_assessment:
            logging.warning("Skipping: Evaluation failed.")
            return None
        
        evaluation_result = {
            "final_assessment": final_assessment,
            "has_source_essay": source_essay is not None
        }

        return evaluation_result


def evaluate_essays(essays: List[Dict], config: AgentConfig, api_key, source_essay, multi_agent) -> List[Dict]:
    logger.info(f"Evaluating {len(essays)} essays")
    
    if multi_agent:
        system = MultiAgentEssayEvaluationSystem(
            api_key=api_key,
            config=config
        )
        
        evaluation_results = []
        
        for i, data in enumerate(tqdm(essays, desc=f"Evaluating essays")):
            
            essay = data['essay']
            essay_prompt = data['prompt_text']
            
            evaluation_result = system.evaluate_essay(essay, essay_prompt, source_essay)
            if evaluation_result is None:
                logger.warning(f"Skipping essay {i}: Evaluation failed. Probably because of parssing issue.")
                continue
            
            evaluation_result['metadata'] = {
                'input_data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            evaluation_results.append(evaluation_result)

    else:
        system = SingleAgentEssayEvaluationSystem(
            api_key=api_key,
            config=config
        )
        
        evaluation_results = []
        
        for i, data in enumerate(tqdm(essays, desc=f"Evaluating essays")):
            
            essay = data['essay']
            essay_prompt = data['prompt_text']
            
            evaluation_result = system.evaluate_essay(essay, essay_prompt, source_essay)
            if evaluation_result is None:
                logger.warning(f"Skipping essay {i}: Evaluation failed. Probably because of parssing issue.")
                continue
            
            evaluation_result['metadata'] = {
                'input_data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            evaluation_results.append(evaluation_result)
            
    return evaluation_results