import json
import os
from google import genai
from google.genai import types

def generate_sar_report(json_file="suspicious_sar_payload.json"):
    # 1. Check if the JSON actually exists so we don't crash
    if not os.path.exists(json_file):
        print("Error: missing 'suspicious_sar_payload.json'.")
        return

    # 2. Read the evidence your XGBoost model dug up
    with open(json_file, 'r') as f:
        evidence = json.load(f)
        
    # 3. The Brainwashing Instructions
    sys_instruct = (
        "You are an expert Anti-Money Laundering (AML) Investigator. "
        "Write a formal FinCEN Suspicious Activity Report (SAR) narrative based on the provided JSON evidence. "
        "Structure exactly like this: INTRODUCTION, BODY (Who, What, When, Where, Why, How), and CONCLUSION. "
        "Write in plain, objective, and professional language. No fluff, no tables, no bullet points. "
        "If a specific detail like a real name is missing, refer to the entity by their transaction ID."
    )
    
    # 4. Wake up Gemini
    try:
        client = genai.Client()
    except Exception as e:
        print("API Key missing! Set your GOOGLE_API_KEY environment variable in the terminal.")
        return
        
    prompt = f"Here is the forensic data. Write the SAR:\n\n{json.dumps(evidence, indent=2)}"
    
    # 5. Generate the report (Temperature 0.2 keeps it boring and factual)
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=sys_instruct,
            temperature=0.2, 
        ),
    )
    
    print("\n" + "="*70)
    print("      OFFICIAL SUSPICIOUS ACTIVITY REPORT (SAR) NARRATIVE")
    print("="*70)
    print(response.text)
    print("="*70)
    
    # Save it so you can put it in your dissertation and graduate
    with open("final_sar_narrative.txt", "w") as f:
        f.write(response.text)
    print("\n Saved the masterpiece to 'final_sar_narrative.txt'. You are officially done.")

if __name__ == "__main__":
    generate_sar_report()