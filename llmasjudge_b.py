import json
from openai import OpenAI
from tqdm import tqdm

# Set your OpenAI API key directly in the client initialization
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

def judge_pair(question, gold_answer, your_answer):
    prompt = f"""
You are an expert judge in computer science. For the given question, two answers are provided (A and B). Your task is to decide which answer is better, or if they are equally good, based on correctness, completeness, and clarity.

Question:
{question}

Answer A (CSBench Gold):
{gold_answer}

Answer B (Your Model):
{your_answer}

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

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Load CSBench test set (JSON, not JSONL)
csbench_data = load_json("test.json")
your_answers = load_jsonl("your_answers.jsonl")

# Create a mapping from question id to your model's answer
your_answers_dict = {item["id"]: item["answer"] for item in your_answers}

results = []
for q in tqdm(csbench_data):
    qid = q["id"]
    question = q["question"]
    gold_answer = q["answer"]  # field name may vary; check CSBench format
    your_answer = your_answers_dict.get(qid, "")
    winner = judge_pair(question, gold_answer, your_answer)
    results.append({
        "question_id": qid,
        "winner": winner,
        "csbench_answer": gold_answer,
        "your_answer": your_answer
    })

with open("judging_results.jsonl", "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Judging complete. Results saved to judging_results.jsonl.")
