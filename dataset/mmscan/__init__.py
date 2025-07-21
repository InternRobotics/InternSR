# flake8: noqa
from .mmscan import MMScan
try:
    from .evaluator.vg_evaluation import VisualGroundingEvaluator
except:
    print('import VisualGroundingEvaluator Error')
    pass
try:
    from .evaluator.qa_evaluation import QuestionAnsweringEvaluator
except:
    print('import QuestionAnsweringEvaluator Error')
    pass
try:
    from .evaluator.gpt_evaluation import GPTEvaluator
except:
    print('import GPTEvaluator Error')
    pass
