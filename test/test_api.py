# test_api.py
import json, requests

url = "http://localhost:8080/chat/reply"
payload = {
    "session_id": 2,
    # "query": "What's Leo's hobby?",
    # "query": "What's your favorite dish?",
    # "query": "Who is Momo",
    # "query": "I want to book Sagano-yu",
    "query": "What is the capital of Japan?",  # 測試無關問題也ok
    "lang": "en"
}

with requests.post(url, data=json.dumps(payload),
                   headers={"Content-Type": "application/json"},
                   stream=True) as r:
    r.raise_for_status()
    chunks = []  # 用來累積所有內容

    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        text = line
        if text.startswith("data: "):
            text = text[len("data: "):]
        if text.strip() == "[DONE]":
            break

        # ===== 法一：stream輸出 =====
        # print("chunk:", text)

        # ===== 法二：累積全部再印 =====
        chunks.append(text)

    full_output = "".join(chunks)
    print("\n=== FULL OUTPUT ===")
    print(full_output)
