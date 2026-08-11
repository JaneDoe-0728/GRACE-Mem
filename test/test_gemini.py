import os

from google import genai
from google.genai import types

# 設定 API Key
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# 系統提示與上下文
SYSTEM_PROMPT = (
    "You are a concise and accurate assistant. "
    "Use the Retrieved Context. If context is insufficient, use general knowledge, "
    "but prefer retrieved facts. Answer directly."
)

question = "How many doctor's appointments did I go to in March?"

context = """
### Entities
- 2023-03-25 (type=Date) — Start date of the therapy sessions (original phrase: March 25th)
- Schedule_Chest_Xray (type=Event) — Planning to arrange an appointment for a chest X‑ray
- Physical_Therapy_Sessions (type=Event) — Regular physical therapy sessions scheduled twice a week to strengthen leg muscles after an ACL tear, starting on 2023-03-25.
- Planner (type=Product) — A planner suggested as a possible gift.
- Twitter_Analytics (type=Product) — Native analytics tool provided by Twitter for measuring post performance
- Terrarium (type=Product) — A glass container used to grow small plants indoors
- 2023-03-03 (type=Date) — Date of diagnosis originally expressed as “March 3rd”.
- Neurologist_appointment (type=Event) — Appointment with neurologist Dr. Johnson for an EMG test concerning left‑hand numbness.
- Follow_up_appointment (type=Event) — Orthopedic surgeon follow‑up appointment concerning knee healing.
- Treatment plan (type=Concept) — Dr. Smith's recommended care for bronchitis  

relationship<|>User<|>Symptom journal<|>keep a symptom journal to track numbness and other symptoms<|>related_to, needs_to_keep
- Consultation with Dr Smith (type=Event) — Discussion between the user and Dr. Smith about whether a chest X‑ray is needed.
### Relationships
- Neurologist_appointment  -> 2023-04-01  | date of the appointment 
- User  -> Appointment with Dr Patel  | needs_to_schedule 
- People_Reached_Metric  -> Total_Reach  | used_for_calculating 
- User  -> Follow_up_appointment  | had a follow‑up appointment with Dr. Thompson 
- User  -> Puerto Rico  | interested in recommendations for 
- Physical_Therapy_Sessions  -> 2023-03-25  | started on this date 
- Neurologist_appointment  -> Left_hand_numbness  | reason for the appointment 
- User  -> La Table de Sorgues  | requested recommendation for this restaurant 

### Evidence Summary
  • [sid=answer_39900a0a_1:10][score=0.488] User : schedule appointment with Dr. Smith Dr. Johnson discuss numbness left hand. keep symptom journal track numbness other symptoms experiencing. Thanks for advice! 
 Assistant : welcome! glad could help. great taking proactive steps to address numbness left hand. thorough detailed in symptom journal, help Dr. Smith Dr. Johnson understand condition make accurate diagnosis. 
 Don t hesitate reach out more questions or concerns before appointments. support you process. 
, take care of yourself manage bronchitis symptoms. get plenty rest, stay hydrated, follow Dr. Smith ' s treatment plan. 
 got this! taking control of health, be proud of. Keep updated on progress, rooting for you! 
"""

# 組成 Gemini API 內容
system_content = types.Content(
    parts=[types.Part.from_text(text=f"{SYSTEM_PROMPT}\n\n---Retrieved Context---\n{context}\n------------------")]
)

# 2. 建立 user query (Content)
user_content = types.Content(
    parts=[types.Part.from_text(text=f"Question: {question}\n\nAnswer:")]
)

# 呼叫 API
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[system_content, user_content],
    config=types.GenerateContentConfig(
        temperature=0.0
    )
)

# 取回回覆
print(response.text)
