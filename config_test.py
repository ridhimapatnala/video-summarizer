import google.generativeai as genai

genai.configure(api_key="AIzaSyC09qs7bXieI-TZD3OQX0du9_YuqO9ewZA")

for m in genai.list_models():
    print(m.name)
