import json
import re
from datasets import load_dataset

def format_answer(answer_text):
    """
    清洗答案文本，确保最后被包裹在 \boxed{} 中。
    （针对 GSM8K 等数据，提取最终的数字或数学表达式）
    """
    # GSM8K 数据中，最终答案在 #### 之后
    if '####' in answer_text:
        reasoning, final_answer = answer_text.split('####')
        final_answer = final_answer.strip()
        formatted_final = f"\\boxed{{{final_answer}}}"
        return f"{reasoning.strip()}\n\nTherefore, the final answer is {formatted_final}"
    else:
        # 如果不是 GSM8K，做通用后处理
        return answer_text

def prepare_competition_dataset(output_path="train_math.jsonl"):
    """
    准备并构造成比赛符合的提示格式。
    我们必须确保结果输出里带有 \boxed{最终答案}。
    """
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    formatted_data = []
    for item in dataset:
        instruction = item["question"]
        raw_answer = item["answer"]
        
        # 准备类似于 Nemotron 推荐的 prompt 格式
        # 加入系统提示约束输出格式
        system_prompt = "You are a logical reasoning assistant. You must write out your step-by-step reasoning and wrap your final answer in \\boxed{}."
        formatted_answer = format_answer(raw_answer)
        
        # 组装为标准的 ChatML 或 LLaMA/Nemotron Instruction 格式
        record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": formatted_answer}
            ]
        }
        formatted_data.append(record)
    
    # 写入 JSONL
    print(f"Writing {len(formatted_data)} records to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in formatted_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print("Dataset preparation complete.")

if __name__ == "__main__":
    prepare_competition_dataset()
