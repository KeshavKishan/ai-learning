from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def analyse_job_description(jd_text):
    """
    Takes a job description as text.
    Sends it to AI with a structured prompt.
    Returns a clean analysis.
    """

    prompt = f"""
You are ARIA, an expert AI career assistant.
Analyse the job description below and respond in EXACTLY this format — no extra text:

JOB TITLE: [job title here]
COMPANY: [company name here]
LOCATION: [location here]
EXPERIENCE REQUIRED: [years here]
LEVEL: [Entry / Mid / Senior]

REQUIRED SKILLS:
- [skill 1]
- [skill 2]
- [skill 3]
- [add all skills found]

KEY RESPONSIBILITIES:
- [responsibility 1]
- [responsibility 2]
- [add all found]

SUMMARY:
[One clear sentence describing what this role is about]

MATCH TIPS:
[Two sentences on what a candidate should highlight to get this role]

Job Description:
{jd_text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are ARIA, an AI career assistant. Always respond in the exact format requested. Be precise and concise."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def main():
    # Sample job description to test with
    # Later this will be replaced by real user input
    sample_jd = """
    Job Title: AI Engineer
    Company: TechCorp India
    Location: Pune, India (Hybrid)

    We are looking for an AI Engineer to join our growing team.
    You will build and deploy AI-powered applications using LLMs,
    RAG systems, and modern AI frameworks.

    Requirements:
    - 2+ years of experience in Python
    - Experience with LangChain or similar frameworks
    - Knowledge of RAG systems and vector databases
    - FastAPI or similar backend framework
    - Docker and basic cloud deployment
    - Strong prompt engineering skills
    - HuggingFace experience is a plus

    Responsibilities:
    - Build RAG pipelines for document Q&A systems
    - Integrate LLM APIs into production applications
    - Optimise prompts for better AI outputs
    - Deploy AI apps using Docker and cloud services
    - Collaborate with product team on AI features
    """

    print("=" * 55)
    print("         ARIA — JOB DESCRIPTION ANALYSER")
    print("=" * 55)
    print("Analysing job description...")
    print()

    result = analyse_job_description(sample_jd)

    print(result)
    print()
    print("=" * 55)
    print("Analysis complete.")
    print("=" * 55)


# This line means: only run main() if this file is run directly
# If this file is imported by another file, main() won't run automatically
if __name__ == "__main__":
    main()