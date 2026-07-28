import csv
from openai import OpenAI

client = OpenAI(api_key="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")

SYSTEM_PROMPT = (
    "You are a concise and accurate assistant. "
    "Use the Retrieved Context. If context is insufficient, use general knowledge, "
    "but prefer retrieved facts. Answer directly."
)

# ======================
# Helper: build messages
# ======================
def build_messages(system_prompt, context, question):
    return [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n\n"
                f"---Retrieved Context---\n"
                f"{context}\n"
                f"------------------"
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nAnswer:",
        },
    ]

# ======================
# Main: CSV → CSV
# ======================
def run_csv_qa(
    input_csv_path: str,
    output_csv_path: str,
    model: str = "gpt-4o-2024-11-20",
):
    with open(input_csv_path, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["gpt_answer"]

        rows = list(reader)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows):
            question = row["question"]
            context = row["retrieved_context"]

            messages = build_messages(
                system_prompt=SYSTEM_PROMPT,
                context=context,
                question=question,
            )

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,  # QA / counting 任務必須低溫
            )

            answer = response.choices[0].message.content.strip()
            row["gpt_answer"] = answer

            writer.writerow(row)

            print(f"[{idx+1}/{len(rows)}] Done")

# ======================
# Run
# ======================
if __name__ == "__main__":
    run_csv_qa(
        input_csv_path="data/sample0_eval_0122.csv",
        output_csv_path="data/sample0_eval_0122gpt.csv",
    )
