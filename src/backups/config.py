# Configuration
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


class DatasetConfig:
    def __init__(
        self,
        dataset: str = "asap",
        dataset_path: str = "path-to-dastaset"
    ):
        self.dataset = dataset
        self.dataset_path = dataset_path

