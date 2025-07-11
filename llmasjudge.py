import json
from openai import OpenAI
from tqdm import tqdm

# Set your OpenAI API key directly in the client initialization
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

def judge_pair(question, answer_a, answer_b):
    prompt = f"""
You are an expert judge in computer science. For the given question, two answers are provided (A and B). Your task is to decide which answer is better, or if they are equally good, based on correctness, completeness, and clarity.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Respond with one line: "A", "B", or "Equal". Optionally, provide a brief justification.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0
    )
    reply = response.choices[0].message.content
    if reply is None:
        return "Equal"
    reply_upper = reply.strip().upper()
    if reply_upper.startswith("A") and "EQUAL" not in reply_upper:
        return "A"
    elif reply_upper.startswith("B"):
        return "B"
    else:
        return "Equal"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

csbench_data = load_jsonl("csbench_en.jsonl")
your_answers = load_jsonl("your_answers.jsonl")
model_answers = load_jsonl("model_answers.jsonl")

results = []
for q, your_a, model_a in tqdm(zip(csbench_data, your_answers, model_answers), total=len(csbench_data)):
    question = q["question"]
    ans_a = your_a["answer"]
    ans_b = model_a["answer"]
    winner = judge_pair(question, ans_a, ans_b)
    results.append({
        "question_id": q["id"],
        "winner": winner,
        "your_answer": ans_a,
        "model_answer": ans_b
    })

with open("judging_results.jsonl", "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Judging complete. Results saved to judging_results.jsonl.")
