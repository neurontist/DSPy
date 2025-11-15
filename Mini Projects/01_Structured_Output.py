from typing import Literal

from dotenv import load_dotenv
import dspy

# load the GEMINI API KEY
load_dotenv(override=True)

# Configure DSPy
lm = dspy.LM('gemini/gemini-2.5-flash', temperature=0.7, max_tokens=6000)
dspy.settings.configure(lm=lm)

# -------------- Define the task signature --------------------
class SupportEmail(dspy.Signature):
    email: str = dspy.InputField()
    subject: str = dspy.OutputField(desc="Subject line of mail")
    priority: Literal["low", "medium", "high"] = dspy.OutputField()
    product: str = dspy.OutputField(
        desc="The product(s) reference. This should be empty string if not known."
    )
    negative_sentiment: bool = dspy.OutputField(desc="True/False")

extract_ticket = dspy.Predict(SupportEmail)

sample_emails = [
    """
    Subject: Screen cracked after one week!

    Hi team,
    I purchased the AlphaTab 11 tablet last Monday and the glass already shattered.
    I’m extremely disappointed and need a replacement ASAP.
    Best,
    Carla
    """,
    """
    Subject: Subscription renewal question

    Hello,
    My CloudSync Pro plan renewed today and I’d like to switch to monthly billing.
    Could you advise?
    Thanks,
    Raj
    """,
    """
    Subject: Your #1 fan

    Loving your alpha tabs. How can I buy more?

    xoxo JEFF
    """,
]

def main() -> None:
    for email in sample_emails:
        pred = extract_ticket(email=email.strip())
        print("\n---Extract Ticket----")
        print(pred)
    print("\n---DSPy History---")
    print(dspy.inspect_history())

if __name__ == '__main__':
    main()

