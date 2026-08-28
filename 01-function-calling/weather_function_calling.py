"""Minh hoạ FUNCTION CALLING thuần với Google Gemini SDK.

Tool `get_weather` được định nghĩa schema thủ công VÀ thực thi ngay trong
chính file app này. Model chỉ QUYẾT ĐỊNH gọi tool nào; app mới là nơi chạy.

Cách chạy:
    source ../.venv/bin/activate
    python weather_function_calling.py   # tự nạp GEMINI_API_KEY từ .env ở gốc repo
"""

from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

client = genai.Client()

MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý thời tiết thân thiện, trả lời bằng tiếng Việt tự nhiên. "
    "Dùng emoji phù hợp (🌧️ 🌤️ 💨 💧). "
    "Tóm tắt ngắn gọn, dễ hiểu, và đưa ra lời khuyên thực tế "
    "(ví dụ: mang ô, mặc áo mỏng, ...)."
)

# 1. App tự định nghĩa schema của tool
get_weather_declaration = types.FunctionDeclaration(
    name="get_weather",
    description="Lấy thời tiết hiện tại của một thành phố",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "city": types.Schema(
                type=types.Type.STRING, description="Tên thành phố"
            )
        },
        required=["city"],
    ),
)

# 1b. App tự định nghĩa schema của tool dự báo thời tiết
get_forecast_declaration = types.FunctionDeclaration(
    name="get_forecast",
    description="Lấy dự báo thời tiết nhiều ngày tới của một thành phố",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "city": types.Schema(
                type=types.Type.STRING, description="Tên thành phố"
            ),
            "days": types.Schema(
                type=types.Type.INTEGER,
                description="Số ngày muốn dự báo (1-7), mặc định 3",
            ),
        },
        required=["city"],
    ),
)

TOOLS = [
    types.Tool(
        function_declarations=[get_weather_declaration, get_forecast_declaration]
    )
]


# 2. App tự thực thi tool (trong thực tế sẽ gọi API thời tiết thật)
def get_weather(city: str) -> str:
    """Trả về thời tiết (mock) của *city*. Dùng làm tool cho model."""
    mock_data = {
        "Hà Nội": {
            "nhiệt_độ": "29°C",
            "thời_tiết": "trời mưa nhẹ",
            "độ_ẩm": "82%",
            "gió": {"hướng": "Đông Nam", "tốc_độ": "12 km/h"},
        },
        "Hồ Chí Minh": {
            "nhiệt_độ": "33°C",
            "thời_tiết": "mưa rào",
            "độ_ẩm": "75%",
            "gió": {"hướng": "Tây Nam", "tốc_độ": "15 km/h"},
        },
        "Đà Nẵng": {
            "nhiệt_độ": "30°C",
            "thời_tiết": "nhiều mây",
            "độ_ẩm": "78%",
            "gió": {"hướng": "Đông", "tốc_độ": "10 km/h"},
        },
    }
    import json

    default = {"nhiệt_độ": "28°C", "thời_tiết": "không có dữ liệu chi tiết"}
    return json.dumps({"city": city, **mock_data.get(city, default)}, ensure_ascii=False)


# 2b. App tự thực thi tool dự báo (trong thực tế sẽ gọi API thời tiết thật)
def get_forecast(city: str, days: int = 3) -> str:
    """Trả về dự báo thời tiết (mock) *days* ngày tới của *city*. Dùng làm tool cho model."""
    import json
    from datetime import date, timedelta

    base_data = {
        "Hà Nội": {"nhiệt_độ_nền": 29, "điều_kiện": ["mưa nhẹ", "nhiều mây", "nắng gián đoạn"]},
        "Hồ Chí Minh": {"nhiệt_độ_nền": 33, "điều_kiện": ["mưa rào", "nắng nóng", "oi bức"]},
        "Đà Nẵng": {"nhiệt_độ_nền": 30, "điều_kiện": ["nhiều mây", "nắng đẹp", "gió nhẹ"]},
    }
    info = base_data.get(
        city, {"nhiệt_độ_nền": 28, "điều_kiện": ["không có dữ liệu chi tiết"]}
    )

    days = max(1, min(days, 7))
    today = date.today()
    du_bao = []
    for i in range(days):
        dieu_kien = info["điều_kiện"][i % len(info["điều_kiện"])]
        nhiet_do = info["nhiệt_độ_nền"] + (i % 3) - 1
        du_bao.append(
            {
                "ngày": (today + timedelta(days=i)).isoformat(),
                "nhiệt_độ": f"{nhiet_do}°C",
                "thời_tiết": dieu_kien,
            }
        )

    return json.dumps({"city": city, "dự_báo": du_bao}, ensure_ascii=False)


AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "get_forecast": get_forecast,
}


def run(prompt: str) -> str:
    """Gửi *prompt* tới Gemini, tự động xử lý function calling và trả về câu trả lời cuối."""
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]

    # 3. Gọi model — model quyết định có gọi tool hay không
    resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            tools=TOOLS,
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    # 4. Vòng lặp: nếu model yêu cầu tool, app TỰ THỰC THI rồi đưa kết quả trả lại
    while resp.function_calls:
        # Thêm phản hồi của model vào lịch sử hội thoại
        contents.append(resp.candidates[0].content)

        function_responses = []
        for fc in resp.function_calls:
            print(f"  [model yêu cầu] {fc.name}({fc.args})")
            func = AVAILABLE_FUNCTIONS[fc.name]
            result = func(**fc.args)  # <-- app chạy, không phải model
            print(f"  [app thực thi]  -> {result}")
            function_responses.append(
                types.Part.from_function_response(
                    name=fc.name, response={"result": result}
                )
            )

        # Gửi kết quả tool trả về cho model
        contents.append(types.Content(role="user", parts=function_responses))
        resp = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=TOOLS,
                system_instruction=SYSTEM_INSTRUCTION,
            ),
        )

    # 5. Model tổng hợp câu trả lời cuối
    return resp.text


if __name__ == "__main__":
    question = "Thời tiết Hà Nội hôm nay thế nào, và dự báo 3 ngày tới ở Đà Nẵng ra sao?"
    print(f"User: {question}\n")
    print("Trả lời:", run(question))
