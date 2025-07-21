import json
from argparse import ArgumentParser
from dataset.mmscan import QuestionAnsweringEvaluator, GPTEvaluator

def evaluate(result_dict):
    results = []
    for q_id in result_dict:
        result = result_dict[q_id]
        result["ID"] = q_id
        results.append(result)
    my_evaluator = QuestionAnsweringEvaluator(show_results=True)
    my_evaluator.update(results)
    metric_dict = my_evaluator.start_evaluation()
    return metric_dict
    
if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--answer-file',
                        required=True)
    args = parser.parse_args()
    answer_data = json.load(open(args.result_json))
    evaluate(answer_data)