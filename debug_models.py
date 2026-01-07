import google.generativeai as genai

genai.configure(api_key="AIzaSyBZzX0Z0sNcBo10E6ppg0quKaHWGaLxTIo")

models = genai.list_models()

print("MODELS FOUND:")
for m in models:
    print(m.name, m.supported_generation_methods)
