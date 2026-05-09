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


def read_jd_from_file(file_path):
    """
    Reads a job description from a .txt file.
    Returns the text content as a string.
    """
    # Check if the file exists before trying to open it
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        print("Make sure your .txt file is in the same folder as this script.")
        return None

    # Open the file, read all text, close it automatically
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    return content

def analyse_skill_gap(jd_analysis, candidate_skills):
    """
    Takes the JD analysis result and candidate's skills.
    Compares them and returns a skill gap report.
    """

    prompt = f"""
You are ARIA, an expert AI career assistant.
A candidate wants to apply for a job. Compare their skills against the job requirements.
Respond in EXACTLY this format — no extra text:

MATCHING SKILLS:
- [skills the candidate already has that match the job]

MISSING SKILLS:
- [skills required by job that candidate does not have]

PRIORITY LEARNING ORDER:
1. [most important missing skill] — [why it's most important] — [estimated weeks to learn basics]
2. [second most important] — [why] — [estimated weeks]
3. [continue for all missing skills]

OVERALL MATCH SCORE: [X out of 10]

HONEST ASSESSMENT:
[Two sentences — how ready is this candidate for this role right now and what is the single most important thing they should do]

Job Requirements Analysis:
{jd_analysis}

Candidate's Current Skills:
{candidate_skills}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are ARIA, an AI career assistant. Be honest and specific. Always respond in the exact format requested."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

def main():
    print("=" * 55)
    print("         ARIA — JOB DESCRIPTION ANALYSER")
    print("=" * 55)

    # Ask user which file to analyse
    file_path = input("Enter the JD filename (e.g. sample_jd.txt): ").strip()

    # Read the file
    jd_text = read_jd_from_file(file_path)

    # If file not found, stop here
    if jd_text is None:
        return

    print()
    print("Analysing job description...")
    print()

    # Send to AI
    # Send to AI — Step 1: analyse the JD
    jd_result = analyse_job_description(jd_text)

    print(jd_result)
    print()
    print("=" * 55)

    # Step 2: ask for candidate skills
    print()
    print("Now let's check your skill match.")
    print("Enter your current skills separated by commas.")
    print("Example: Python, FastAPI, Docker, SQL, REST APIs")
    print()
    candidate_skills = input("Your skills: ").strip()

    print()
    print("Analysing skill gap...")
    print()

    # Step 3: run skill gap analysis
    gap_result = analyse_skill_gap(jd_result, candidate_skills)

    print("=" * 55)
    print("         ARIA — SKILL GAP ANALYSIS")
    print("=" * 55)
    print(gap_result)
    print()
    print("=" * 55)
    print("Analysis complete.")
    print("=" * 55)


# This line means: only run main() if this file is run directly
# If this file is imported by another file, main() won't run automatically
if __name__ == "__main__":
    main()