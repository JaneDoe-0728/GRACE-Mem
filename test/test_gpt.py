from openai import OpenAI

client = OpenAI(api_key="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")

SYSTEM_PROMPT = (
    "You are a concise and accurate assistant. "
    "Use the Retrieved Context. If context is insufficient, use general knowledge, "
    "but prefer retrieved facts. Answer directly."
)

question = "I'm looking back at our previous chess game and I was wondering, what was the move you made after 27. Kg2 Bd5+?"

context = """
=== Entities ===
- Chess game (Event): A chess game discussed on 2023-05-21 that includes a full move sequence from the opening (1.e4 c5) through to 27...Bd5+, featuring moves such as Be6 which attacks the rook and gains control of the e‑file.
- Kg2 Bd5+ (Activity): The legal move played on move 27, moving the king to g2 and delivering check with bishop to d5.
- g4 (Activity): The assistant suggested the move g4 on move 25 to gain space on the kingside and restrict the opponent's pawn structure.
- Chess game move 1. d4 (Activity): The opening move “d4”, a pawn advance to the d4 square played by White, is a common chess opening.
- Black (Person): The player controlling the black pieces in the described chess game.
- Rh8 (Activity): A chess move notation indicating a piece moving to the square h8.
- Current chess position (Event): The ongoing chess game state after move 26...Rh8, with White to move.
- Chess game between User and Assistant (Event): A chess match between the user (White) and the assistant (Black) covering moves from the opening 1.e4 c5 through to the final position after 29.Rd3 Rh4, with a full move sequence documented up to at least 27…Bd5+.
- Parenting blogs (Product): Websites where authors share personal experiences and tips about parenting.
- Classic Cappuccino (Topic): A coffee recipe consisting of espresso, steamed milk, and frothed milk.
- Italy (Location): Region in Europe where nationalist movements emerged under the influence of the French Revolution.
- rook on h8 (Concept): A rook already occupying the h8 square, making the move Rh8 illegal.
- Assistant (Person): An AI assistant that responds in the conversation, providing answers throughout the dialogue and also acting as Black in the described chess game.

=== Relationships ===
- User -> Kg2 Bd5+: user requested the legal move Kg2 Bd5+
- User -> Chess game move 1. d4: User made the opening move in the chess game
- User -> hxg4: User made the chess move hxg4; User asks about the chess move hxg4
- Chess game between User and Assistant -> User: User plays as White in the chess game
- Current chess position -> White: the ongoing game state involves the White player
- 7.1 virtual surround sound -> Rocket League: surround sound impact may be less noticeable for this game
- Chess game between User and Assistant -> Assistant: Assistant plays as Black in the chess game
- Assistant -> Chess game move 1. d4: assistant makes the chess move d4
- Current chess position -> Black: the ongoing game state involves the Black player
- User -> Featured snippet: User aims to improve content shown in the featured snippet
- Exclusive Pokémon -> Pokémon Shield: Some Exclusive Pokémon are only in Pokémon Shield
- g4 -> Chess game between User and Assistant: assistant's suggested move in the ongoing chess match
- Kg2 Bd5+ -> Chess game: legal move played in the ongoing chess game

### Evidence Summary
  • [2023/05/21 (Sun) 13:30][sid=answer_sharegpt_d6JJiqH_76:10][score=0.590] . c5. Nxd4. Bd3. O e5. Nc3 Nxc3. 
 bxc3 exd4. cxd4 Bb4. Rb1. Re1 Bxe1. Qxe1. Ba3. 
 Rxe1 Be6. Bd6 Kd7. Bxb8. Rc1 Rhc8. a3 h5. dxc4. Bxc4 
. Re1 Rc7. Re5 f6. Re3 Rbc8. h3 g5. g4 hxg4. hxg4 Rh8 
!.? 
  • [2023/05/21 (Sun) 13:30][sid=answer_sharegpt_d6JJiqH_76:2][score=0.569] . c5. cxd4. Nxd4. Bd3 d5. O e5. Nc3 Nxc3. 
 bxc3 exd4. cxd4 Bb4. Rb1 a5. Re1 Bxe1. Qxe1 Qe7. Ba3. 
. Bd6 Kd7. Bxb8. Rc1 Rhc8. a3 h5. dxc4. Bxc4 
. Re1 Rc7. Re5 f6. Re3 Rbc8. h3 g5 
! 
. g4 
 pawn structure.? 
  • [2023/05/21 (Sun) 13:30][sid=answer_sharegpt_d6JJiqH_76:20][score=0.559] pawn 
. Nf6. cxd4. Nxd4. Bd3 d5. O e5. Nc3 Nxc3. 
. cxd4 Bb4. Rb1. Re1. Ba3. 
 Rxe1. Bd6 Kd7. Bxb8. Rc1 Rhc8. a3 h5. dxc4. Bxc4 
. Re1 Rc7. Re5 f6. Re3 Rbc8. h3 g5. g4 hxg4. hxg4 Rh8. 
 Kg2 Bd5 
. Kg3. 
  • [2023/05/21 (Sun) 13:30][sid=answer_sharegpt_d6JJiqH_76:24][score=0.546] . 
. Nf6. cxd4. Nxd4. Bd3 d5. O e5. Nc3 Nxc3. 
. cxd4 Bb4. Rb1. Re1. Ba3. 
 Rxe1 Be6. Bd6 Kd7. Bxb8. Rc1 Rhc8. a3 h5. dxc4. Bxc4 
. Re1 Rc7. Re5 f6. Re3 Rbc8. h3 g5. hxg4 Rh8. 
 Kg2 Bd5. Kg3 Be6 
. Rd3 
  • [2023/05/21 (Sun) 13:30][sid=answer_sharegpt_d6JJiqH_76:16][score=0.538] . Nf3 Nf6. cxd4. Nxd4. Bd3 d5. O e5. Nc3 Nxc3. 
 bxc3 exd4. cxd4 Bb4. Rb1. Re1 Bxe1. Ba3. 
 Rxe1. Bd6 Kd7. Bxb8. Rc1 Rhc8. a3 h5. dxc4. Bxc4 
. Re1 Rc7. Re5 f6. Re3 Rbc8. h3 g5. g4 hxg4. hxg4 Rh8 
,. 
. Kg2 Bd5 
  • [2023/05/21 (Sun) 13:30][sid=answer_sharegpt_d6JJiqH_76:22][score=0.531] Be6 
,. 
. c5. Nf3 Nf6. cxd4. Nxd4. Bd3 d5. O e5. Nc3 Nxc3. bxc3 exd4. cxd4 Be6. Rb1. Nc6. Bd6 Kd7. Bxb8. Rc1 Rhc8. a3 h5. dxc4. Bxc4. Re1 Rc7. Re5 f6. Re3 Rbc8. h3 g5. hxg4. hxg4 Rh8. Kg2 Bd5. Kg3 
"""

messages = [
    {
        "role": "system",
        "content": (
            f"{SYSTEM_PROMPT}\n\n"
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

response = client.chat.completions.create(
    model="gpt-4o-2024-11-20",
    messages=messages,
    temperature=0.0,  # RAG 建議低溫
)

answer = response.choices[0].message.content
print(answer)
