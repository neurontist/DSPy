from typing import Literal

from dotenv import load_dotenv
import dspy

# load the GEMINI API KEY
load_dotenv(override=True)

# Configure DSPy
lm = dspy.LM('gemini/gemini-2.5-flash', temperature=0.7, max_tokens=6000)
dspy.settings.configure(lm=lm)

# -------------- Define the task signature --------------------
class LoanRisk(dspy.Signature):
    """Check the risk of givingt the loan and whether to approve it or not"""
    applicant_profile: str = dspy.InputField()
    loan_risk: Literal["low","medium","high"] = dspy.OutputField()
    approved: bool = dspy.OutputField(desc="approve this person for loan?")

risk_checker = dspy.ChainOfThought(LoanRisk)

sample_profile = """Name: Jane Diaz
Credit score: 612
Annual income: $84k
Existing debt: $55k
Requested amount: $25k
Loan purpose: consolidate credit cards
"""

def main() -> None:
    pred = risk_checker(applicant_profile=sample_profile)

    print("\n--------- DSPy History ------")
    print(dspy.inspect_history())

    print("\n--------- Prediction --------")
    print(pred)

if __name__ == '__main__':
    main()