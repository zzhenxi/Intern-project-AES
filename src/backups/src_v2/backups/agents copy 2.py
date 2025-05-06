import os
import json
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Configuration
class EssayEvaluationConfig:
    def __init__(
        self,
        n_agents: int = 4,
        max_score: int = 10,
        min_score: int = 1,
        max_holistic_score: int = 100,
        min_holistic_score: int = 0,
        feedback: bool = True,
        model_name: str = "gpt-4.1-mini"
    ):
        self.n_agents = n_agents
        self.max_score = max_score
        self.min_score = min_score
        self.max_holistic_score = max_holistic_score
        self.min_holistic_score = min_holistic_score
        self.feedback = feedback
        self.model_name = model_name


class PersonaAgent:
    """Agent that generates a specialized evaluator persona based on essay and prompt"""
    
    def __init__(self, config: EssayEvaluationConfig):
        self.config = config
        self.llm = ChatOpenAI(model=config.model_name, temperature=0.7)
        
    def generate_personas(self, essay: str, essay_prompt: str) -> List[Dict[str, str]]:
        persona_template = """
        You are an expert at creating specialized personas for evaluating essays.
        
        Given the following essay and its prompt, create {n_agents} distinct evaluator personas.
        
        These personas MUST include experts focused on:
        1. Essay structure and grammar/mechanics
        2. Content and subject matter
        3. Alignment with the original prompt requirements
        4. The purpose/style of the essay (e.g., persuasive, narrative, etc.)
        
        Essay Prompt:
        {essay_prompt}
        
        Essay:
        {essay}
        
        For each persona, provide:
        1. A name
        2. Professional background
        3. Specific area of expertise
        4. Evaluation focus
        
        Format your response as a JSON array of persona objects with keys: "name", "background", "expertise", "focus"
        Make sure your response can be parsed by Python's json.loads() function.
        """
        
        persona_prompt = ChatPromptTemplate.from_template(persona_template)
        persona_chain = (
            {"essay_prompt": RunnablePassthrough(), "essay": RunnablePassthrough(), "n_agents": lambda _: self.config.n_agents}
            | persona_prompt
            | self.llm
        )
        
        result = persona_chain.invoke({"essay_prompt": essay_prompt, "essay": essay})
        response_text = result.content
        
        # Extract JSON from response text - looking for the first JSON-like structure
        try:
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: try to parse the entire response
                return json.loads(response_text)
        except json.JSONDecodeError:
            # If parsing fails, return a default set of personas
            print("Warning: Failed to parse JSON response from persona generation. Using default personas.")
            return [
                {
                    "name": "Dr. Emily Wordsworth",
                    "background": "PhD in English Language and Literature",
                    "expertise": "Writing mechanics and structure",
                    "focus": "Essay structure, grammar, and mechanics"
                },
                {
                    "name": "Prof. Martin Chen",
                    "background": "Subject matter specialist",
                    "expertise": "Content evaluation and factual accuracy",
                    "focus": "Essay content and subject matter relevance"
                },
                {
                    "name": "Dr. James Rodriguez",
                    "background": "Prompt design specialist",
                    "expertise": "Prompt adherence and requirement fulfillment",
                    "focus": "Alignment with prompt requirements"
                },
                {
                    "name": "Prof. Diana Kim",
                    "background": "Rhetorical studies expert",
                    "expertise": "Essay purpose and style evaluation",
                    "focus": "Essay purpose and stylistic elements"
                }
            ]


class RubricAgent:
    """Agent that generates evaluation rubrics based on personas"""
    
    def __init__(self, config: EssayEvaluationConfig):
        self.config = config
        self.llm = ChatOpenAI(model=config.model_name, temperature=0.5)
        
    def generate_rubrics(self, personas: List[Dict[str, str]], essay: str, essay_prompt: str) -> List[Dict[str, Any]]:
        rubrics = []
        
        rubric_template = """
        You are {name}, {background} with expertise in {expertise}.
        
        Your task is to create a detailed evaluation rubric for assessing an essay. The rubric should focus on your specific area of expertise: {focus}.
        
        Essay Prompt:
        {essay_prompt}
        
        Essay to Evaluate:
        {essay}
        
        Create a rubric with 3-5 specific traits that evaluate aspects of the essay within your area of expertise.
        
        For each trait:
        - Provide a clear name
        - Give a detailed description of what this trait measures
        - Include specific criteria for different score levels within the range of {min_score} (lowest) to {max_score} (highest)
        
        Format your response as a JSON object with the following structure:
        {{
            "persona": {{
                "name": "Your persona name",
                "focus": "Your area of focus"
            }},
            "traits": [
                {{
                    "name": "Name of trait",
                    "description": "Description of trait",
                    "criteria": [
                        {{
                            "score": score_value,
                            "description": "What this score means"
                        }},
                        ...more score criteria...
                    ]
                }},
                ...more traits...
            ]
        }}
        
        Make sure your response can be parsed by Python's json.loads() function.
        """
        
        for persona in personas:
            rubric_prompt = ChatPromptTemplate.from_template(rubric_template)
            rubric_chain = (
                {
                    "name": lambda _: persona["name"],
                    "background": lambda _: persona["background"],
                    "expertise": lambda _: persona["expertise"],
                    "focus": lambda _: persona["focus"],
                    "essay_prompt": RunnablePassthrough(),
                    "essay": RunnablePassthrough(),
                    "min_score": lambda _: self.config.min_score,
                    "max_score": lambda _: self.config.max_score
                }
                | rubric_prompt
                | self.llm
            )
            
            result = rubric_chain.invoke({"essay_prompt": essay_prompt, "essay": essay})
            response_text = result.content
            
            # Extract JSON from response text
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    rubric = json.loads(json_str)
                    rubrics.append(rubric)
                else:
                    # Fallback: try to parse the entire response
                    rubric = json.loads(response_text)
                    rubrics.append(rubric)
                    
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON response for rubric generation for {persona['name']}. Skipping.")
            
        return rubrics


class ScoringAgent:
    """Agent that scores essays based on persona-specific rubrics"""
    
    def __init__(self, config: EssayEvaluationConfig):
        self.config = config
        self.llm = ChatOpenAI(model=config.model_name, temperature=0.2)
        
    def generate_scores(self, rubrics: List[Dict[str, Any]], essay: str, essay_prompt: str) -> List[Dict[str, Any]]:
        all_scores = []
        
        scoring_template = """
        You are {persona_name}, focusing on evaluating essays from the perspective of {persona_focus}.
        
        Your task is to evaluate the following essay according to your specialized rubric.
        
        Essay Prompt:
        {essay_prompt}
        
        Essay to Evaluate:
        {essay}
        
        Your evaluation rubric has the following traits:
        {traits_json}
        
        For each trait in your rubric:
        1. First, provide detailed reasoning for your evaluation
        2. Then, assign a score within the range {min_score} to {max_score}
        
        Format your response as a JSON object with the following structure:
        {{
            "persona": {{
                "name": "Your persona name",
                "focus": "Your area of focus"
            }},
            "trait_scores": [
                {{
                    "trait_name": "Name of trait",
                    "rationale": "Detailed explanation of your reasoning",
                    "score": assigned_score
                }},
                ...more trait scores...
            ]
        }}
        
        Make sure your response can be parsed by Python's json.loads() function.
        """
        
        for rubric in rubrics:
            persona_name = rubric["persona"]["name"]
            persona_focus = rubric["persona"]["focus"]
            traits_json = json.dumps(rubric["traits"])
            
            scoring_prompt = ChatPromptTemplate.from_template(scoring_template)
            scoring_chain = (
                {
                    "persona_name": lambda _: persona_name,
                    "persona_focus": lambda _: persona_focus,
                    "essay_prompt": RunnablePassthrough(),
                    "essay": RunnablePassthrough(),
                    "traits_json": lambda _: traits_json,
                    "min_score": lambda _: self.config.min_score,
                    "max_score": lambda _: self.config.max_score
                }
                | scoring_prompt
                | self.llm
            )
            
            result = scoring_chain.invoke({"essay_prompt": essay_prompt, "essay": essay})
            response_text = result.content
            
            # Extract JSON from response text
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    scores = json.loads(json_str)
                    all_scores.append(scores)
                else:
                    # Fallback: try to parse the entire response
                    scores = json.loads(response_text)
                    all_scores.append(scores)
                    
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON response for scoring from {persona_name}. Skipping.")
            
        return all_scores


class MetaScoreAgent:
    """Agent that aggregates scores from multiple evaluators into a final assessment"""
    
    def __init__(self, config: EssayEvaluationConfig):
        self.config = config
        self.llm = ChatOpenAI(model=config.model_name, temperature=0.3)
        
    def generate_meta_score(self, all_scores: List[Dict[str, Any]], essay: str, essay_prompt: str) -> Dict[str, Any]:
        meta_template = """
        You are a Meta Evaluator responsible for synthesizing multiple expert evaluations into a coherent final assessment.
        
        Your task is to review evaluations from {n_agents} expert personas and produce a comprehensive final score and assessment.
        
        Essay Prompt:
        {essay_prompt}
        
        Essay:
        {essay}
        
        Expert Evaluations:
        {evaluations_json}
        
        Please analyze these evaluations to:
        1. Identify unique evaluation traits across all personas
        2. Determine appropriate weight for each trait based on its importance
        3. Calculate a final holistic score within the range of {min_holistic_score} to {max_holistic_score}
        4. {feedback_instruction}
        
        Format your response as a JSON object with the following structure:
        {{
            "trait_summary": [
                {{
                    "trait": "Name of trait",
                    "focus": "Related focus area",
                    "score": normalized_score,
                    "weight": assigned_weight
                }},
                ...more traits...
            ],
            "holistic_score": final_score,
            "feedback": "Comprehensive feedback" // Only if feedback is required
        }}
        
        Make sure your response can be parsed by Python's json.loads() function.
        """
        
        feedback_instruction = "Provide comprehensive feedback with strengths and areas for improvement" if self.config.feedback else "No feedback needed"
        
        meta_prompt = ChatPromptTemplate.from_template(meta_template)
        meta_chain = (
            {
                "n_agents": lambda _: self.config.n_agents,
                "essay_prompt": RunnablePassthrough(),
                "essay": RunnablePassthrough(),
                "evaluations_json": lambda _: json.dumps(all_scores),
                "min_holistic_score": lambda _: self.config.min_holistic_score,
                "max_holistic_score": lambda _: self.config.max_holistic_score,
                "feedback_instruction": lambda _: feedback_instruction
            }
            | meta_prompt
            | self.llm
        )
        
        result = meta_chain.invoke({"essay_prompt": essay_prompt, "essay": essay})
        response_text = result.content
        
        # Extract JSON from response text
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: try to parse the entire response
                return json.loads(response_text)
                
        except json.JSONDecodeError:
            print("Warning: Failed to parse JSON response from meta scoring. Returning default assessment.")
            # Return a default assessment as fallback
            return {
                "trait_summary": [
                    {
                        "trait": "Overall Quality",
                        "focus": "General Assessment",
                        "score": (self.config.max_holistic_score - self.config.min_holistic_score) / 2,
                        "weight": 1.0
                    }
                ],
                "holistic_score": (self.config.max_holistic_score - self.config.min_holistic_score) / 2,
                "feedback": "The system encountered an error while generating detailed feedback."
            }


