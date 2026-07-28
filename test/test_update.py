# test_update.py
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import httpx

from AI.KG.pipeline import generate_llm_entity_ops

# 載入環境變數
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)
LLM_API    = os.getenv("LLM_API")
MODEL_NAME = os.getenv("MODEL_NAME")

# 建 client 丟給 kg_pipeline 用
client = OpenAI(base_url=LLM_API, api_key="", http_client=httpx.Client(timeout=15.0))

# 測試資料
# === new_entities（生活化 + 同概念不同名稱）===
new_entities = [
    {
        "entity_name": "Mom",
        "entity_type": "person",
        "entity_description": "User's mother; lives in Taipei; surname Lin."
    },
    {
        "entity_name": "BILLY 書櫃",
        "entity_type": "product",
        "entity_description": "IKEA bookshelf, white, about 80x28x202 cm."
    },
    {
        "entity_name": "Blue IKEA shopping bag",
        "entity_type": "product",
        "entity_description": "Large blue polypropylene bag used for shopping."
    },
    {
        "entity_name": "Coriander",
        "entity_type": "food",
        "entity_description": "Herb; used in salads and soups."
    },
    {
        "entity_name": "Formosa",
        "entity_type": "location",
        "entity_description": "Historical name for Taiwan."
    },
    {
        "entity_name": "Shea Butter Hand Moisturizer",
        "entity_type": "product",
        "entity_description": "A generic shea-butter hand moisturizer; travel-friendly tube."
    },
    {
        "entity_name": "Chinese New Year",
        "entity_type": "event",
        "entity_description": "The lunar new year festival celebrated in many Asian regions."
    }
]

# === similar_map（每個 key = (name, type)；每項 3 個候選）===
similar_map = {
    # 1) Mom —— 預期 UPDATE（Mom ≈ Angela Lin）
    ("Mom", "person"): [
        (
            {
                "id": "person_angela_lin_1970",
                "name": "Angela Lin",
                "type": "person",
                "description": "Mother of the user; resides in Taipei."
            },
            0.90
        ),
        (
            {
                "id": "person_angelina_lin",
                "name": "Angelina Lin",
                "type": "person",
                "description": "Entertainer; different individual."
            },
            0.66
        ),
        (
            {
                "id": "person_father_lin",
                "name": "Mr. Lin",
                "type": "person",
                "description": "User's father."
            },
            0.58
        )
    ],

    # 2) BILLY 書櫃 —— 預期 UPDATE（中文別名 ≈ IKEA BILLY Bookcase）
    ("BILLY 書櫃", "product"): [
        (
            {
                "id": "product_ikea_billy_bookcase_white",
                "name": "IKEA BILLY Bookcase (White)",
                "type": "product",
                "description": "80x28x202 cm variant in white."
            },
            0.92
        ),
        (
            {
                "id": "product_ikea_kallax_shelf_unit",
                "name": "IKEA KALLAX Shelf Unit",
                "type": "product",
                "description": "Different product line of shelving."
            },
            0.71
        ),
        (
            {
                "id": "category_bookcases",
                "name": "Bookcases",
                "type": "category",
                "description": "Furniture category for bookshelves."
            },
            0.55
        )
    ],

    # 3) Blue IKEA shopping bag —— 預期 UPDATE（口語描述 ≈ FRAKTA）
    ("Blue IKEA shopping bag", "product"): [
        (
            {
                "id": "product_ikea_frakta_bag",
                "name": "IKEA FRAKTA Bag",
                "type": "product",
                "description": "Iconic large blue polypropylene shopping bag."
            },
            0.93
        ),
        (
            {
                "id": "product_canvas_tote_bag_generic",
                "name": "Canvas Tote Bag",
                "type": "product",
                "description": "Generic unbranded canvas tote."
            },
            0.60
        ),
        (
            {
                "id": "category_shopping_accessories",
                "name": "Shopping Accessories",
                "type": "category",
                "description": "Totes, carts, and accessories."
            },
            0.57
        )
    ],

    # 4) Coriander —— 預期 UPDATE（學名/別名：Coriander ≈ Cilantro）
    ("Coriander", "food"): [
        (
            {
                "id": "food_cilantro_coriander",
                "name": "Cilantro",
                "type": "food",
                "description": "Also known as coriander; herb used in many cuisines."
            },
            0.94
        ),
        (
            {
                "id": "food_parsley",
                "name": "Parsley",
                "type": "food",
                "description": "Different herb, often confused with cilantro."
            },
            0.62
        ),
        (
            {
                "id": "category_herbs",
                "name": "Herbs",
                "type": "category",
                "description": "Culinary herbs category."
            },
            0.56
        )
    ],

    # 5) Formosa —— 預期 UPDATE（歷史別名 ≈ Taiwan）
    ("Formosa", "location"): [
        (
            {
                "id": "location_taiwan",
                "name": "Taiwan",
                "type": "location",
                "description": "Island in East Asia; also historically called Formosa."
            },
            0.95
        ),
        (
            {
                "id": "location_tainan_city",
                "name": "Tainan",
                "type": "location",
                "description": "A city in southern Taiwan."
            },
            0.68
        ),
        (
            {
                "id": "category_islands",
                "name": "Islands",
                "type": "category",
                "description": "General category of islands."
            },
            0.54
        )
    ],

    # 6) Shea Butter Hand Moisturizer —— 預期 ADD（泛稱 ≠ 特定品牌實體）
    ("Shea Butter Hand Moisturizer", "product"): [
        (
            {
                "id": "product_loccitane_shea_hand_cream",
                "name": "L'OCCITANE Shea Hand Cream",
                "type": "product",
                "description": "Shea butter hand cream 30ml."
            },
            0.73
        ),
        (
            {
                "id": "product_nivea_repair_hand_cream",
                "name": "NIVEA Repair Hand Cream",
                "type": "product",
                "description": "Hand cream for very dry hands."
            },
            0.64
        ),
        (
            {
                "id": "category_hand_moisturizers",
                "name": "Hand Moisturizers",
                "type": "category",
                "description": "Category for hand creams and lotions."
            },
            0.58
        )
    ],

    # 7) Chinese New Year —— 預期 UPDATE（別名 ≈ Lunar New Year）
    ("Chinese New Year", "event"): [
        (
            {
                "id": "event_lunar_new_year",
                "name": "Lunar New Year",
                "type": "event",
                "description": "New year festival based on the lunar calendar."
            },
            0.96
        ),
        (
            {
                "id": "event_mid_autumn_festival",
                "name": "Mid-Autumn Festival",
                "type": "event",
                "description": "Moon festival in the 8th lunar month."
            },
            0.66
        ),
        (
            {
                "id": "category_traditional_festivals",
                "name": "Traditional Festivals",
                "type": "category",
                "description": "Category for traditional festivals."
            },
            0.55
        )
    ]
}


if __name__ == "__main__":
    results = generate_llm_entity_ops(new_entities, similar_map)
    print(json.dumps(results, indent=2, ensure_ascii=False))
