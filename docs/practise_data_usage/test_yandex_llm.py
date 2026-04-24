import openai

YANDEX_CLOUD_FOLDER = "b1gm7......t0kst15tq"
YANDEX_CLOUD_API_KEY = 'AQVN3XbuZ2......vEpRc_Se5L4EQ'

YANDEX_CLOUD_MODEL = "deepseek-v32/latest"
# YANDEX_CLOUD_MODEL = "gpt://b1gm7vfr214t0kst15tq/llama3.1-70b-instruct/latest"

client = openai.OpenAI(
  api_key=YANDEX_CLOUD_API_KEY,
  base_url="https://ai.api.cloud.yandex.net/v1",
  project=YANDEX_CLOUD_FOLDER
)

response = client.responses.create(
#   model=f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}",
model='gpt://b1gm7vfr214t0kst15tq/llama3.1-70b-instruct/latest',
  temperature=0.3,
  instructions="",
  input="Когда будет конец света?",
  max_output_tokens=500
)

print(response.output_text)