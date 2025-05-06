import os
import json
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Configuration
class EssayEvaluationConfig:
    def __init__(
        self,import os
import json
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import JsonOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

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
        grade_level: str = "high school",
        essay_type: str = "argumentative",
        model_name: str = "gpt-4o"
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
    
    def __init__(self, config: EssayEvaluationConfig):
        self.config = config
        self.llm = ChatOpenAI(model=config.model_name, temperature=0.7)
        
    def generate_personas(self, essay: str, essay_prompt: str) -> List[Dict[str, str]]:
        persona_template = """
        You are an expert at creating specialized personas for evaluating essays.
        
        Given the following essay and its prompt, create {n_agents} distinct evaluator personas.
        
        Important Context:
        - Grade Level: {grade_level} 
        - Essay Type: {essay_type}
        
        These personas MUST include experts focused on:
        1. Essay structure and grammar/mechanics
        2. Content and subject matter
        3. Alignment with the original prompt requirements
        4. The specific requirements of {essay_type} essays
        
        Each persona should have expertise appropriate for evaluating {grade_level} level writing.
        
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
        
        persona_prompt = ChatPromptTemplate.from_template(persona_template)
        persona_chain = (
            {
                "essay_prompt": RunnablePassthrough(), 
                "essay": RunnablePassthrough(), 
                "n_agents": lambda _: self.config.n_agents,
                "grade_level": lambda _: self.config.grade_level,
                "essay_type": lambda _: self.config.essay_type
            }
            | persona_prompt
            | self.llm
            | JsonOutputParser()
        )
        
        return persona_chain.invoke({"essay_prompt": essay_prompt, "essay": essay})


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
        
        Important Context:
        - Grade Level: {grade_level} 
        - Essay Type: {essay_type}
        
        Your rubric should be specifically calibrated for {grade_level} students writing {essay_type} essays.
        
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
                    "max_score": lambda _: self.config.max_score,
                    "grade_level": lambda _: self.config.grade_level,
                    "essay_type": lambda _: self.config.essay_type
                }
                | rubric_prompt
                | self.llm
                | JsonOutputParser()
            )
            
            rubric = rubric_chain.invoke({"essay_prompt": essay_prompt, "essay": essay})
            rubrics.append(rubric)
            
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
        
        Important Context:
        - Grade Level: {grade_level} 
        - Essay Type: {essay_type}
        
        Remember to calibrate your expectations and evaluation to what is appropriate for {grade_level} students writing {essay_type} essays.
        
        Essay Prompt:
        {essay_prompt}
        
        Essay to Evaluate:
        {essay}
        
        Your evaluation rubric has the following traits:
        {traits_json}
        
        For each trait in your rubric:
        1. First, provide detailed reasoning for your evaluation, considering the student's grade level
        2. Then, assign a score within the range {min_score} to {max_score}
        3. Ensure your assessment accounts for what is developmentally appropriate for {grade_level} students
        
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
            
            scoring_prompt = ChatPromptTemplate.from_template(scoring_template)
            scoring_chain = (
                {
                    "persona_name": lambda _: persona_name,
                    "persona_focus": lambda _: persona_focus,
                    "essay_prompt": RunnablePassthrough(),
                    "essay": RunnablePassthrough(),
                    "traits_json": lambda _: traits_json,
                    "min_score": lambda _: self.config.min_score,
                    "max_score": lambda _: self.config.max_score,
                    "grade_level": lambda _: self.config.grade_level,
                    "essay_type": lambda _: self.config.essay_type
                }
                | scoring_prompt
                | self.llm
                | JsonOutputParser()
            )
            
            scores = scoring_chain.invoke({"essay_prompt": essay_prompt, "essay": essay})
            all_scores.append(scores)
            
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
        
        Important Context:
        - Grade Level: {grade_level} 
        - Essay Type: {essay_type}
        
        Your final assessment should reflect appropriate expectations for {grade_level} students writing {essay_type} essays.
        
        Essay Prompt:
        {essay_prompt}
        
        Essay:
        {essay}
        
        Expert Evaluations:
        {evaluations_json}
        
        Please analyze these evaluations to:
        1. Identify unique evaluation traits across all personas
        2. Determine appropriate weight for each trait based on its importance for {essay_type} essays at the {grade_level} level
        3. Calculate a final holistic score within the range of {min_holistic_score} to {max_holistic_score}
        4. {feedback_instruction} (Ensure feedback is developmentally appropriate and actionable for {grade_level} students)
        
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
        
        meta_prompt = ChatPromptTemplate.from_template(meta_template)
        meta_chain = (
            {
                "n_agents": lambda _: self.config.n_agents,
                "essay_prompt": RunnablePassthrough(),
                "essay": RunnablePassthrough(),
                "evaluations_json": lambda _: json.dumps(all_scores),
                "min_holistic_score": lambda _: self.config.min_holistic_score,
                "max_holistic_score": lambda _: self.config.max_holistic_score,
                "feedback_instruction": lambda _: feedback_instruction,
                "grade_level": lambda _: self.config.grade_level,
                "essay_type": lambda _: self.config.essay_type
            }
            | meta_prompt
            | self.llm
            | JsonOutputParser()
        )
        
        return meta_chain.invoke({"essay_prompt": essay_prompt, "essay": essay})


class EssayEvaluationSystem:
    """Main system that orchestrates the multi-agent evaluation process"""
    
    def __init__(self, api_key: str = None, config: Optional[EssayEvaluationConfig] = None):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            
        self.config = config or EssayEvaluationConfig()
        self.persona_agent = PersonaAgent(self.config)
        self.rubric_agent = RubricAgent(self.config)
        self.scoring_agent = ScoringAgent(self.config)
        self.meta_agent = MetaScoreAgent(self.config)
        
    def evaluate_essay(self, essay: str, essay_prompt: str) -> Dict[str, Any]:
        """Full evaluation pipeline that returns structured assessment of an essay"""
        
        # Step 1: Generate specialized personas
        personas = self.persona_agent.generate_personas(essay, essay_prompt)
        
        # Step 2: Generate rubrics for each persona
        rubrics = self.rubric_agent.generate_rubrics(personas, essay, essay_prompt)
        
        # Step 3: Score essay using each persona's rubric
        all_scores = self.scoring_agent.generate_scores(rubrics, essay, essay_prompt)
        
        # Step 4: Generate meta-assessment with final score
        final_assessment = self.meta_agent.generate_meta_score(all_scores, essay, essay_prompt)
        
        # Compile complete evaluation result
        evaluation_result = {
            "personas": personas,
            "rubrics": rubrics,
            "detailed_scores": all_scores,
            "final_assessment": final_assessment
        }
        
        return evaluation_result


# Example usage
if __name__ == "__main__":
    # Configuration
    config = EssayEvaluationConfig(
        n_agents=4,
        max_score=5,
        min_score=1,
        max_holistic_score=100,
        min_holistic_score=0,
        feedback=True,
        model_name="gpt-4o"
    )
    
    # Initialize system with OpenAI API key
    system = EssayEvaluationSystem(
        api_key="your_openai_api_key_here",  # Replace with your API key
        config=config
    )
    
    # Sample essay and prompt
    essay_prompt = "Discuss the impact of artificial intelligence on modern education."
    
    essay = """
    Artificial Intelligence: Reshaping Education
    
    The integration of artificial intelligence in education represents one of the most significant technological shifts in modern pedagogy. As AI systems become more sophisticated, they are transforming how students learn and how educators teach.
    
    Personalized learning stands as perhaps the most promising application of AI in education. Traditional classroom models often struggle to address the diverse needs of many students simultaneously. AI-powered platforms can analyze individual student performance, identify knowledge gaps, and adapt content delivery to match each student's learning pace and style. This personalization helps struggling students receive the additional support they need while allowing advanced learners to progress at an accelerated rate.
    
    Assessment is another area where AI demonstrates valuable potential. Automated grading systems can evaluate objective assignments instantaneously, freeing educators from time-consuming tasks and providing students with immediate feedback. More sophisticated AI tools are beginning to assess complex work like essays, analyzing factors including structure, argumentation, and coherence. While not replacing human judgment, these systems offer preliminary evaluations that help teachers manage large class loads more effectively.
    
    Administrative efficiency also improves with AI implementation. Institutions are deploying AI to streamline enrollment processes, schedule classes, and manage resources. These administrative applications allow educational institutions to operate more efficiently, potentially redirecting resources toward improving educational quality.
    
    However, the integration of AI in education raises important concerns. The digital divide may widen as schools with greater resources adopt advanced AI tools while underfunded institutions fall further behind. Questions about data privacy emerge as AI systems collect extensive information about students' learning behaviors and personal characteristics. Additionally, overreliance on technology might diminish crucial human elements of education—particularly the mentorship, inspiration, and emotional intelligence that skilled teachers provide.
    
    In conclusion, AI holds transformative potential for education while presenting significant challenges. The most successful educational futures will likely be those that thoughtfully integrate AI capabilities with irreplaceable human guidance, creating balanced learning environments that leverage technological advantages while preserving the essential human connections that give education its deepest value.
    """
    
    # Evaluate essay
    evaluation_result = system.evaluate_essay(essay, essay_prompt)
    
    # Print formatted results
    print(json.dumps(evaluation_result, indent=2))
        n_agents: int = 4,
        max_score: int = 10,
        min_score: int = 1,
        max_holistic_score: int = 100,
        min_holistic_score: int = 0,
        feedback: bool = True,
        grade_level: str = "",
        essay_type: str = "",
        model_name: str = "gpt-4.1-mini"
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


