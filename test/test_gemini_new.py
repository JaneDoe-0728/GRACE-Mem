import csv
import time
from google import genai
from google.genai import types, errors

client = genai.Client(api_key="AIzaSyDlcoYRiUERPlIwAjbQbEQhnJvo8NDpNdg")

SYSTEM_PROMPT = (
    "You are a concise and accurate assistant. "
    "Use the Retrieved Context. If context is insufficient, use general knowledge, "
    "but prefer retrieved facts. Answer directly."
)

def build_contents(system_prompt, context, question):
    return [
        types.Content(parts=[
            types.Part.from_text(
                text=(
                    f"{system_prompt}\n\n"
                    f"---Retrieved Context---\n"
                    f"{context}\n"
                    f"------------------"
                )
            )
        ]),
        types.Content(parts=[
            types.Part.from_text(
                text=f"Question: {question}\n\nAnswer:"
            )
        ]),
    ]

def call_gemini_safe(model, contents, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.0),
            )
        except errors.ServerError as e:
            # 503 overloaded
            retry_after = 30
            try:
                retry_after = int(
                    e.response_json["error"]["details"][-1]["retryDelay"]
                    .replace("s", "")
                )
            except Exception:
                pass

            print(f"⚠️ 503 overloaded，等待 {retry_after}s 後重試（{attempt+1}/{max_retries}）")
            time.sleep(retry_after)

    raise RuntimeError("多次重試仍失敗")

def run_csv_qa(input_csv_path, output_csv_path, model="gemini-2.5-flash"):
    with open(input_csv_path, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["gemini_answer"]
        rows = list(reader)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows):
            question = row["question"]
            context = row["Retrieved_Context"]

            contents = build_contents(SYSTEM_PROMPT, context, question)

            response = call_gemini_safe(model, contents)
            answer = response.text.strip()

            row["gemini_answer"] = answer
            writer.writerow(row)

            print(f"[{idx+1}/{len(rows)}] Done")

            # ⭐ 核心：強制限速
            time.sleep(2.5)

if __name__ == "__main__":
    run_csv_qa(
        "data/multi_incorrect_judged.csv",
        "data/multi_incorrect_gemini_output.csv",
    )
