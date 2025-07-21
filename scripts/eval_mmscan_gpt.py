import json
from argparse import ArgumentParser
from dataset.mmscan import GPTEvaluator

def evaluate(result_dict,eval_size,api_key,tmp_path):
    results = []
    for q_id in result_dict:
        result = result_dict[q_id]
        result["ID"] = q_id
        results.append(result)
    gpt_evaluator = GPTEvaluator(eval_size=eval_size, api_key=api_key)
    metric_dict = gpt_evaluator.load_and_eval(results,num_threads=5, tmp_path=tmp_path)
    return metric_dict
    
if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--answer-file',
                        required=True)
    parser.add_argument('--eval_size',
                        default=1000)
    parser.add_argument('--api_key',
                        required=True)
    parser.add_argument('--tmp_path',
                        required=True)
    args = parser.parse_args()
    answer_data = json.load(open(args.result_json))
    evaluate(answer_data,args.eval_size,args.api_key,args.tmp_path)