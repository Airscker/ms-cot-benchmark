import os
import json
import re

# Claude loader: loads all answer fields from claude-3.5-sonnet-20241022
def load_claude_answers(benchmark_dir='benchmark/claude-3.5-sonnet-20241022'):
    files = [f for f in os.listdir(benchmark_dir) if f.startswith('batch_') and f.endswith('.json')]
    files.sort(key=lambda x: int(re.search(r'batch_(\d+)', x).group(1)))
    answers = []
    for fname in files:
        with open(os.path.join(benchmark_dir, fname), 'r') as f:
            data = json.load(f)
            for result in data.get('results', []):
                # Prefer direct answer field if present
                if 'analysis' in result:
                    answers.append(result['analysis'])
                else:
                    answers.append('')
    return answers

# Llama loader: loads all answer fields from llama3-70B-instruct and Meta-Llama-3-8B-Instruct
# Accepts a list of directories
def load_llama_answers(benchmark_dir='benchmark/llama3-70B-instruct'):
    answers = []
    files = [f for f in os.listdir(benchmark_dir) if f.startswith('response_') and f.endswith('.json')]
    files.sort(key=lambda x: int(re.search(r'response_(\d+)', x).group(1)))
    for fname in files:
        with open(os.path.join(benchmark_dir, fname), 'r') as f:
            data = json.load(f)
            # Prefer direct answer field if present
            try:
                message = data['response']['choices'][0]['message']
                if 'content' in message:
                    answers.append(message['content'])
                else:
                    answers.append('')
            except Exception:
                continue
    return answers

# GPT-4o loader: loads all answer fields from gpt-4o-mini with proper sorting
def load_gpt4o_answers(benchmark_dir='benchmark/gpt-4o-mini'):
    files = [f for f in os.listdir(benchmark_dir) if f.startswith('batch_') and f.endswith('.jsonl')]
    
    # Sort files by batch_size * batch_id (logical order)
    def sort_key(filename):
        match = re.search(r'batch_(\d+)_(\d+)', filename)
        if match:
            batch_size = int(match.group(1))
            batch_id = int(match.group(2))
            return batch_size * batch_id
        return 0
    
    files.sort(key=sort_key)
    
    # Track processed custom_ids to avoid duplicates
    processed_ids = set()
    answers = []
    
    for fname in files:
        with open(os.path.join(benchmark_dir, fname), 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        custom_id = data.get('custom_id', '')
                        
                        # Skip if we've already processed this custom_id
                        if custom_id in processed_ids:
                            continue
                        
                        processed_ids.add(custom_id)
                        
                        # Extract answer from response
                        if 'response' in data and 'body' in data['response']:
                            body = data['response']['body']
                            if 'choices' in body and len(body['choices']) > 0:
                                message = body['choices'][0].get('message', {})
                                if 'content' in message:
                                    answers.append(message['content'])
                                else:
                                    answers.append('')
                            else:
                                answers.append('')
                        else:
                            answers.append('')
                    except json.JSONDecodeError:
                        continue
    return answers

if __name__ == '__main__':
    claude_answers = load_claude_answers()
    print(len(claude_answers), claude_answers[0], '' in claude_answers)
    gpt4o_answers = load_gpt4o_answers()
    print(len(gpt4o_answers), gpt4o_answers[0], '' in gpt4o_answers)
    llama_3_8b = load_llama_answers(benchmark_dir='benchmark/Meta-Llama-3-8B-Instruct')
    print(len(llama_3_8b), llama_3_8b[10], '' in llama_3_8b)
    llama_70b = load_llama_answers(benchmark_dir='benchmark/llama3-70B-instruct')
    print(len(llama_70b), llama_70b[10], '' in llama_70b)
    # save all answers to separate files
    with open('benchmark/claude_answers.json', 'w') as f:
        json.dump(claude_answers, f)
    with open('benchmark/gpt4o_answers.json', 'w') as f:
        json.dump(gpt4o_answers, f)
    with open('benchmark/llama_3_8b_answers.json', 'w') as f:
        json.dump(llama_3_8b, f)
    with open('benchmark/llama_70b_answers.json', 'w') as f:
        json.dump(llama_70b, f)
    
