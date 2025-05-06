import os, logging
import json
from typing import List, Dict, Any, Optional, Literal
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

logger = logging.getLogger(__name__)

class AgentConfig:
    def __init__(
        self,
        n_agents: int = 4,
        max_score: int = 10,
        min_score: int = 1,
        max_holistic_score: int = 100,
        min_holistic_score: int = 0,
        feedback: bool = True,
        grade_level: str = "high school",
        essay_type: str = "argumentative",
        model_name: str = "gpt-4.1-mini",
        api_type: Literal["openai", "anthropic"] = "openai" 
    ):
        self.n_agents = n_agents
        self.max_score = max_score
        self.min_score = min_score
        self.max_holistic_score = max_holistic_score
        self.min_holistic_score = min_holistic_score
        self.feedback = feedback
        self.grade_level = grade_level
        self.essay_type = essay_type
        self.model_name = model_name


class PersonaAgent:
    """Agent that generates a specialized evaluator persona based on essay and prompt"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        if config.api_type == "anthropic":
            self.llm = ChatAnthropic(model=config.model_name, temperature=0.7)
        else:
            self.llm = ChatOpenAI(model=config.model_name, temperature=0.7)
        
    def generate_personas(self, essay: str, essay_prompt: str, source_essay: Optional[str] = None) -> List[Dict[str, str]]:
        persona_template = """
        You are an expert at creating specialized personas for evaluating essays.
        
        Given the following essay and its prompt, create {n_agents} distinct evaluator personas.
        
        These personas MUST include experts focused on:
        1. Essay structure and grammar/mechanics
        2. Content and subject matter
        3. Alignment with the original prompt requirements
        4. The specific requirements of {essay_type} essays
        5. Each persona should have expertise appropriate for evaluating {grade_level} level writing.
        {source_essay_requirement}
        
        {source_essay_section}
        
        Essay Prompt:
        {essay_prompt}
        
        Essay:
        {essay}
        
        For each persona, provide:
        1. A name
        2. Professional background (relevant to {grade_level} education)
        3. Specific area of expertise
        4. Evaluation focus
        
        Format your response as a JSON array of persona objects with keys: "name", "background", "expertise", "focus"
        """
        
        # Add Source Essay-specific requirements if Source Essay is provided
        source_essay_requirement = ""
        source_essay_section = ""
        
        if source_essay:
            source_essay_requirement = "6. At least one persona should focus specifically on how well the essay incorporates, references, or builds upon the provided source essay."
            source_essay_section = f"""
            Source Essay (which the student's essay should reference or build upon):
            {source_essay}
            """
        
        persona_prompt = ChatPromptTemplate.from_template(persona_template)
        persona_chain = (
            {
                "essay_prompt": RunnablePassthrough(), 
                "essay": RunnablePassthrough(),
                "source_essay_requirement": lambda _: source_essay_requirement,
                "source_essay_section": lambda _: source_essay_section,
                "n_agents": lambda _: self.config.n_agents,
                "grade_level": lambda _: self.config.grade_level,
                "essay_type": lambda _: self.config.essay_type
            }
            | persona_prompt
            | self.llm
        )

        result = persona_chain.invoke({"essay_prompt": essay_prompt, "essay": essay})
        response_text = result.content        

        try:
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: try to parse the entire response
                return json.loads(response_text)
        except:
            logging.error("Error parsing JSON response from persona generation.")
            return None
        

class RubricAgent:
    """Agent that generates evaluation rubrics based on personas"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        if config.api_type == "anthropic":
            self.llm = ChatAnthropic(model=config.model_name, temperature=0.5)
        else:
            self.llm = ChatOpenAI(model=config.model_name, temperature=0.5)
        
    def generate_rubrics(self, personas: List[Dict[str, str]], essay: str, essay_prompt: str, source_essay: Optional[str] = None) -> List[Dict[str, Any]]:
        rubrics = []
        
        rubric_template = """
        You are {name}, {background} with expertise in {expertise}.
        
        Your task is to create a detailed evaluation rubric for assessing an essay. The rubric should focus on your specific area of expertise: {focus}.
        Your rubric should be specifically calibrated for {grade_level} students writing {essay_type} essays.
        
        {source_essay_section}

        Essay Prompt:
        {essay_prompt}
        
        Essay to Evaluate:
        {essay}
        
        Create a rubric with 3-5 specific traits that evaluate aspects of the essay within your area of expertise.
        
        For each trait:
        - Provide a clear name 
        - Give a detailed description of what this trait measures
        - Include specific criteria for different score levels within the range of {min_score} (lowest) to {max_score} (highest)
        - Ensure criteria are appropriate for {grade_level} expectations and the conventions of {essay_type} essays
        
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
        """
        
        for persona in personas:
            # Add Source Essay-specific instructions if Source Essay is provided
            source_essay_section = ""
            
            if source_essay:
                source_essay_section = f"""
                Source Essay (which the essay should reference or build upon):
                {source_essay}
                """
            
            rubric_prompt = ChatPromptTemplate.from_template(rubric_template)
            rubric_chain = (
                {
                    "name": lambda _: persona["name"],
                    "background": lambda _: persona["background"],
                    "expertise": lambda _: persona["expertise"],
                    "focus": lambda _: persona["focus"],
                    "essay_prompt": RunnablePassthrough(),
                    "essay": RunnablePassthrough(),
                    "source_essay_section": lambda _: source_essay_section,
                    "min_score": lambda _: self.config.min_score,
                    "max_score": lambda _: self.config.max_score,
                    "grade_level": lambda _: self.config.grade_level,
                    "essay_type": lambda _: self.config.essay_type
                }
                | rubric_prompt
                | self.llm
            )
            
            result = rubric_chain.invoke({"essay_prompt": essay_prompt, "essay": essay})
            response_text = result.content
            
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
                    
            except:
                logging.error(f"Error parsing JSON response from persona {persona['name']}'s rubric generation.")
                return None
            
        return rubrics


class ScoringAgent:
    """Agent that scores essays based on persona-specific rubrics"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        if config.api_type == "anthropic":
            self.llm = ChatAnthropic(model=config.model_name, temperature=0.2)
        else:
            self.llm = ChatOpenAI(model=config.model_name, temperature=0.2)
        
    def generate_scores(self, rubrics: List[Dict[str, Any]], essay: str, essay_prompt: str, source_essay: Optional[str] = None) -> List[Dict[str, Any]]:
        all_scores = []
        
        scoring_template = """
        You are {persona_name}, focusing on evaluating essays from the perspective of {persona_focus}.
        
        Your task is to evaluate the following essay according to your specialized rubric.
        Remember to calibrate your expectations and evaluation to what is appropriate for {grade_level} students writing {essay_type} essays.
        
        {source_essay_section}
        
        Essay Prompt:
        {essay_prompt}
        
        Essay to Evaluate:
        {essay}
        
        Your evaluation rubric has the following traits:
        {traits_json}
        
        For each trait in your rubric:
        1. First, provide detailed reasoning for your evaluation, considering the student's grade level
        2. Then, assign a score within the range {min_score} to {max_score}
        {source_essay_instruction}
        
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
        """
        
        for rubric in rubrics:
            persona_name = rubric["persona"]["name"]
            persona_focus = rubric["persona"]["focus"]
            traits_json = json.dumps(rubric["traits"])
            
            # Add Source Essay-specific sections if Source Essay is provided
            source_essay_section = ""
            source_essay_instruction = ""
            
            if source_essay:
                source_essay_section = f"""
                Source Essay (which the essay should reference or build upon):
                {source_essay}
                """
                
                # Add special instructions for source-focused personas
                if "source" in persona_focus.lower() or "reference" in persona_focus.lower():
                    source_essay_instruction = """3. How well the essay integrates or responds to the source essay material"""
            
            scoring_prompt = ChatPromptTemplate.from_template(scoring_template)
            scoring_chain = (
                {
                    "persona_name": lambda _: persona_name,
                    "persona_focus": lambda _: persona_focus,
                    "essay_prompt": RunnablePassthrough(),
                    "essay": RunnablePassthrough(),
                    "source_essay_section": lambda _: source_essay_section,
                    "source_essay_instruction": lambda _: source_essay_instruction,
                    "traits_json": lambda _: traits_json,
                    "min_score": lambda _: self.config.min_score,
                    "max_score": lambda _: self.config.max_score,
                    "grade_level": lambda _: self.config.grade_level,
                    "essay_type": lambda _: self.config.essay_type
                }
                | scoring_prompt
                | self.llm
            )

            result = scoring_chain.invoke({"essay_prompt": essay_prompt, "essay": essay})
            response_text = result.content
            
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
                    
            except:
                logging.error(f"Error parsing JSON response from persona {persona_name}'s scoring generation.")
                return None
            
        return all_scores


class MetaScoreAgent:
    """Agent that aggregates scores from multiple evaluators into a final assessment"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        if config.api_type == "anthropic":
            self.llm = ChatAnthropic(model=config.model_name, temperature=0.3)
        else:
            self.llm = ChatOpenAI(model=config.model_name, temperature=0.3)
        
    def generate_meta_score(self, all_scores: List[Dict[str, Any]], essay: str, essay_prompt: str, source_essay: Optional[str] = None) -> Dict[str, Any]:
        meta_template = """
        You are a Meta Evaluator responsible for synthesizing multiple expert evaluations into a coherent final assessment.
        
        Your task is to review evaluations from {n_agents} expert personas and produce a comprehensive final score and assessment.
        Your final assessment should reflect appropriate expectations for {grade_level} students writing {essay_type} essays.
        
        Essay Prompt:
        {essay_prompt}
        
        {source_essay_section}
        
        Essay:
        {essay}
        
        Expert Evaluations:
        {evaluations_json}
        
        Please analyze these evaluations to:
        1. Identify unique evaluation traits across all personas
        2. Determine appropriate weight for each trait based on its importance for {essay_type} essays at the {grade_level} level
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
        """
        
        feedback_instruction = "Provide comprehensive feedback with strengths and areas for improvement" if self.config.feedback else "No feedback needed"
        
        # Add Source Essay-specific sections if Source Essay is provided
        source_essay_section = ""
        
        if source_essay:
            source_essay_section = f"""
            Source Essay (which the essay should reference or build upon):
            {source_essay}
            """
        
        meta_prompt = ChatPromptTemplate.from_template(meta_template)
        meta_chain = (
            {
                "n_agents": lambda _: self.config.n_agents,
                "essay_prompt": RunnablePassthrough(),
                "essay": RunnablePassthrough(),
                "source_essay_section": lambda _: source_essay_section,
                "evaluations_json": lambda _: json.dumps(all_scores),
                "min_holistic_score": lambda _: self.config.min_holistic_score,
                "max_holistic_score": lambda _: self.config.max_holistic_score,
                "feedback_instruction": lambda _: feedback_instruction,
                "grade_level": lambda _: self.config.grade_level,
                "essay_type": lambda _: self.config.essay_type
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
                
        except:
            logging.error(f"Error parsing JSON response from meta scoring generation.")
            return None

class SingleEvalAgent:
    """A single agent that evaluates essays directly without multiple personas"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        if config.api_type == "anthropic":
            self.llm = ChatAnthropic(model=config.model_name, temperature=0.3)
        else:
            self.llm = ChatOpenAI(model=config.model_name, temperature=0.3)
        
    def generate_holistic_score(self, essay: str, essay_prompt: str, source_essay: Optional[str] = None) -> Dict[str, Any]:
        evaluation_template = """
        You are an expert essay evaluator for {grade_level} level {essay_type} essays.

        Your task is to thoroughly evaluate the following essay based on the prompt provided.
        
        {source_essay_section}
        
        Essay Prompt:
        {essay_prompt}
        
        Essay to Evaluate:
        {essay}
        
        Then, assign a holistic score between {min_holistic_score} and {max_holistic_score} that reflects the overall quality of the essay.
        
        {feedback_instruction}
        
        Format your response as a JSON object with the following structure:
        {{
            "holistic_score": your_holistic_score,
            "feedback": "Comprehensive feedback" // Only if feedback is required
        }}
        """
        
        feedback_instruction = "Provide comprehensive feedback with strengths and areas for improvement" if self.config.feedback else "No feedback needed"
        
        # Add Source Essay-specific sections if Source Essay is provided
        source_essay_section = ""
        source_reference_json = ""
        
        if source_essay:
            source_essay_section = f"""
            Source Essay (which the essay should reference or build upon):
            {source_essay}
            """

        evaluation_prompt = ChatPromptTemplate.from_template(evaluation_template)
        evaluation_chain = (
            {
                "essay_prompt": RunnablePassthrough(),
                "essay": RunnablePassthrough(),
                "source_essay_section": lambda _: source_essay_section,
                "source_reference_json": lambda _: source_reference_json,
                "min_holistic_score": lambda _: self.config.min_holistic_score,
                "max_holistic_score": lambda _: self.config.max_holistic_score,
                "feedback_instruction": lambda _: feedback_instruction,
                "grade_level": lambda _: self.config.grade_level,
                "essay_type": lambda _: self.config.essay_type
            }
            | evaluation_prompt
            | self.llm
        )
        
        result = evaluation_chain.invoke({"essay_prompt": essay_prompt, "essay": essay})
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
                
        except Exception as e:
            logging.error(f"Error parsing JSON response from evaluation: {e}")
            return None

