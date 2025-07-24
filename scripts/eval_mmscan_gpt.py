import json
import pandas as pd
from argparse import ArgumentParser
from dataset.mmscan import GPTEvaluator

def evaluate_json(result_file,eval_size,api_key,tmp_path):
    result_dict = json.load(open(result_file))
    results = []
    for q_id in result_dict:
        result = result_dict[q_id]
        result["ID"] = q_id
        results.append(result)
    gpt_evaluator = GPTEvaluator(eval_size=eval_size, api_key=api_key)
    metric_dict = gpt_evaluator.load_and_eval(results,num_threads=5, tmp_path=tmp_path)
    return metric_dict

def evaluate_xlsx(result_file,eval_size,api_key,tmp_path):
    result_dict = pd.read_excel(result_file)
    results = []
    questions = result_dict["question"].tolist()
    answers = result_dict["answer"].tolist()
    IDs = result_dict["ID"].tolist()
    predictions = result_dict["prediction"].tolist()
    for (question,answer,ID,pred) in zip(questions,answers,IDs,predictions):
        result = {}
        result['question'] = question.split('<ImageHere>')[-1]
        result['gt'] = eval(answer)
        result['pred'] = [pred]
        result['ID'] = ID
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
    if args.answer_file.split('.')[-1]=='json':
        print(evaluate_json(args.answer_file,args.eval_size,args.api_key,args.tmp_path))
    elif args.answer_file.split('.')[-1]=='xlsx':
        print(evaluate_xlsx(args.answer_file,args.eval_size,args.api_key,args.tmp_path))
    else:
        print("Only support the evaluation for xlsx / json files.")