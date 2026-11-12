
import csv

from pathlib import Path


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

QUESTIONS_PATH = EVALUATION_DIR / "evaluation_questions.csv"

RELEVANCE_PATH = EVALUATION_DIR / "relevance_label.csv"


# --------------------------------------------------
# EVALUATION QUESTIONS
# --------------------------------------------------

evaluation_questions = [

    {
        "question_id": "Q001",
        "question": "When did the motor vehicle collision occur?",
        "expected_answer": "The collision occurred on February 11, 2024, at approximately 17:42.",
        "answerable": True,
        "question_type": "factual"
    },

    {
        "question_id": "Q002",
        "question": "What symptoms did the patient report immediately after the collision?",
        "expected_answer": "He reported posterior neck pain, mid and lower back pain, and a jolt or electric sensation down the right leg.",
        "answerable": True,
        "question_type": "clinical"
    },

    {
        "question_id": "Q003",
        "question": "What diagnoses were recorded in the emergency physician's note?",
        "expected_answer": "Acute cervical strain, acute lumbosacral strain, right lumbar radiculopathy, and motor vehicle collision.",
        "answerable": True,
        "question_type": "clinical"
    },

    {
        "question_id": "Q004",
        "question": "What did the cervical spine CT show?",
        "expected_answer": "No acute fracture or traumatic malalignment. Mild chronic degenerative disc disease at C5-C6 was reported.",
        "answerable": True,
        "question_type": "imaging"
    },

    {
        "question_id": "Q005",
        "question": "What findings were reported in the lumbar spine X-ray?",
        "expected_answer": "No acute fracture or listhesis. Degenerative disc disease and facet arthropathy were reported at L4-L5 and L5-S1.",
        "answerable": True,
        "question_type": "imaging"
    },

    {
        "question_id": "Q006",
        "question": "What abnormality was found at L4-L5 on the March 6 lumbar MRI?",
        "expected_answer": "A right paracentral disc protrusion measuring approximately 6 mm, extending into the right lateral recess.",
        "answerable": True,
        "question_type": "imaging"
    },

    {
        "question_id": "Q007",
        "question": "Which nerve root was contacted and mildly displaced by the L4-L5 disc protrusion?",
        "expected_answer": "The traversing right L5 nerve root.",
        "answerable": True,
        "question_type": "clinical"
    },

    {
        "question_id": "Q008",
        "question": "What right-sided neurological abnormalities were documented during the February 19 primary care visit?",
        "expected_answer": "Right extensor hallucis longus strength was 4/5, sensation was decreased over the right L5 dermatome, and the right Achilles reflex was 1+ compared with 2+ on the left.",
        "answerable": True,
        "question_type": "clinical"
    },

    {
        "question_id": "Q009",
        "question": "What was the patient's Oswestry Disability Index score at the initial physical therapy evaluation?",
        "expected_answer": "The Oswestry Disability Index was 48%, classified as severe disability.",
        "answerable": True,
        "question_type": "clinical"
    },

    {
        "question_id": "Q010",
        "question": "When was the documented right L5 transforaminal epidural steroid injection performed?",
        "expected_answer": "April 23, 2024.",
        "answerable": True,
        "question_type": "treatment"
    },

    {
        "question_id": "Q011",
        "question": "What were the patient's pain scores immediately before and after the April 23 injection?",
        "expected_answer": "Pain was 7/10 before the procedure and 2/10 immediately afterward during the anesthetic phase.",
        "answerable": True,
        "question_type": "treatment"
    },

    {
        "question_id": "Q012",
        "question": "What reaction did the patient report after taking the codeine-containing medication?",
        "expected_answer": "He reported an itchy rash across his chest and forearms, which resolved after discontinuing the medication.",
        "answerable": True,
        "question_type": "medication"
    },

    {
        "question_id": "Q013",
        "question": "What gabapentin plan was initiated at the February 19 primary care visit?",
        "expected_answer": "Gabapentin 300 mg at bedtime was added, with instructions to titrate as tolerated.",
        "answerable": True,
        "question_type": "medication"
    },

    {
        "question_id": "Q014",
        "question": "What important differences appear between the signed and unsigned April 9 physical therapy progress notes?",
        "expected_answer": "The signed note reports right leg pain unchanged at 7/10, right great toe extension at 4/5, and a positive right straight leg raise. The unsigned draft reports right leg pain improved to 5/10, great toe extension at 5/5, and a negative straight leg raise. The draft is explicitly marked unsigned.",
        "answerable": True,
        "question_type": "record_conflict"
    },

    {
        "question_id": "Q015",
        "question": "What does the May 21 handwritten-note transcription report about the response to the right transforaminal injection?",
        "expected_answer": "It records [illegible] relief for 10 days, followed by return of leg symptoms. The type or degree of relief cannot be established because the relevant word is illegible.",
        "answerable": True,
        "question_type": "clinical"
    },

    {
        "question_id": "Q016",
        "question": "What side and nerve root were identified in the August 28 EMG/NCS impression?",
        "expected_answer": "Chronic left L5 radiculopathy with active denervation.",
        "answerable": True,
        "question_type": "clinical"
    },

    {
        "question_id": "Q017",
        "question": "What surgical procedure was performed on September 10, 2024?",
        "expected_answer": "Right L4-L5 hemilaminotomy with microdiscectomy and decompression of the right L5 nerve root.",
        "answerable": True,
        "question_type": "treatment"
    },

    {
        "question_id": "Q018",
        "question": "How was the patient's condition described at the November 5 postoperative follow-up?",
        "expected_answer": "Right leg pain was essentially resolved at 0-1/10, motor strength was 5/5 including right EHL, and mild residual sensory reduction remained in the right L5 dermatome.",
        "answerable": True,
        "question_type": "clinical"
    },

    {
        "question_id": "Q019",
        "question": "What previous low back condition was documented in the August 2019 urgent care record?",
        "expected_answer": "Mechanical low back pain for three weeks after moving furniture, with intermittent radiation into the right buttock but no pain below the knee.",
        "answerable": True,
        "question_type": "historical"
    },

    {
        "question_id": "Q020",
        "question": "What previous motor vehicle collision history did the patient disclose during the February 19 visit?",
        "expected_answer": "He reported a prior rear-end collision in 2021, followed by a few weeks of back trouble that resolved without formal treatment.",
        "answerable": True,
        "question_type": "historical"
    },

    {
        "question_id": "Q021",
        "question": "Which medications were dispensed on September 10, 2024, according to the pharmacy history?",
        "expected_answer": "Oxycodone/acetaminophen 5-325 mg, docusate sodium 100 mg, and methocarbamol 750 mg.",
        "answerable": True,
        "question_type": "medication"
    },

    {
        "question_id": "Q022",
        "question": "When did the employer's records indicate that the patient returned to full duty?",
        "expected_answer": "January 6, 2025.",
        "answerable": True,
        "question_type": "administrative"
    },

    {
        "question_id": "Q023",
        "question": "What total charges were listed in the December 9, 2024 itemized billing summary?",
        "expected_answer": "$44,756.00.",
        "answerable": True,
        "question_type": "administrative"
    },

    {
        "question_id": "Q024",
        "question": "Which requested records were listed as not produced in the records custodian certification?",
        "expected_answer": "The cervical MRI report, a legible duplicate of the May 21 handwritten orthopedic note, clinical documentation for the June 11 transforaminal epidural steroid injection, and physical therapy daily notes for visits 15 through 22.",
        "answerable": True,
        "question_type": "document_completeness"
    },

    {
        "question_id": "Q025",
        "question": "What are the complete detailed findings in the cervical MRI report?",
        "expected_answer": "I don't know. The cervical MRI report itself was not produced. The orthopedic consultation only mentions that it showed no significant cord compression.",
        "answerable": False,
        "question_type": "unanswerable"
    },

    {
        "question_id": "Q026",
        "question": "What clinical findings and procedural details were documented for the June 11 transforaminal epidural steroid injection?",
        "expected_answer": "I don't know. The records custodian states that the clinical documentation corresponding to the June 11 injection was not located in this production.",
        "answerable": False,
        "question_type": "unanswerable"
    },

    {
        "question_id": "Q027",
        "question": "What exercises and treatment details were recorded in physical therapy visits 15 through 22?",
        "expected_answer": "I don't know. The physical therapy daily notes for visits 15 through 22 were not included in the produced records.",
        "answerable": False,
        "question_type": "unanswerable"
    },

    {
        "question_id": "Q028",
        "question": "What are the exact words represented by the illegible portions of the May 21 handwritten orthopedic note?",
        "expected_answer": "I don't know. The transcription explicitly marks those portions as [illegible], and a legible duplicate was not produced.",
        "answerable": False,
        "question_type": "unanswerable"
    }
]


# --------------------------------------------------
# RELEVANCE LABELS
# --------------------------------------------------

relevance_labels = [

    # Q001 - Accident date and time
    ("Q001", 1, 3, "EMS report directly records the incident date and time."),

    # Q002 - Immediate symptoms
    ("Q002", 1, 3, "EMS report documents immediate symptoms."),
    ("Q002", 3, 2, "Emergency physician note describes the symptoms in more detail."),

    # Q003 - ED diagnoses
    ("Q003", 3, 3, "Emergency physician note lists the diagnoses."),
    ("Q003", 5, 2, "Discharge summary repeats the discharge diagnoses."),

    # Q004 - Cervical CT
    ("Q004", 4, 3, "Cervical CT report contains the findings and impression."),

    # Q005 - Lumbar X-ray
    ("Q005", 4, 3, "Lumbar radiograph report contains the findings and impression."),

    # Q006 - MRI disc protrusion
    ("Q006", 8, 3, "Lumbar MRI directly describes the right paracentral protrusion."),

    # Q007 - Nerve root
    ("Q007", 8, 3, "Lumbar MRI identifies the traversing right L5 nerve root."),

    # Q008 - Primary care neurological findings
    ("Q008", 6, 3, "Primary care note records strength, sensation and reflex findings."),

    # Q009 - Initial ODI
    ("Q009", 7, 3, "Initial physical therapy evaluation records ODI of 48%."),

    # Q010 - Injection date
    ("Q010", 11, 3, "Procedure note records the injection date and procedure."),

    # Q011 - Injection pain scores
    ("Q011", 11, 3, "Procedure note records pre- and post-procedure pain scores."),

    # Q012 - Codeine reaction
    ("Q012", 6, 3, "Primary care note describes the patient's rash after taking codeine."),
    ("Q012", 2, 2, "ED triage record documents codeine allergy with rash."),
    ("Q012", 5, 1, "Discharge record shows the codeine-containing prescription."),

    # Q013 - Gabapentin plan
    ("Q013", 6, 3, "Primary care plan initiates gabapentin 300 mg at bedtime."),
    ("Q013", 19, 1, "Pharmacy history provides later dispensing context."),

    # Q014 - Conflicting PT notes
    ("Q014", 10, 3, "Signed progress note records persistent right leg pain and unchanged deficit."),
    ("Q014", 12, 3, "Unsigned draft records improved symptoms and examination findings."),

    # Q015 - May 21 handwritten note
    ("Q015", 13, 3, "Transcription contains the partially illegible injection response statement."),

    # Q016 - EMG laterality
    ("Q016", 14, 3, "EMG/NCS impression identifies chronic left L5 radiculopathy."),

    # Q017 - Surgery
    ("Q017", 15, 3, "Operative report records the procedure and date."),

    # Q018 - Postoperative outcome
    ("Q018", 17, 3, "November 5 postoperative visit records recovery findings."),

    # Q019 - 2019 history
    ("Q019", 18, 3, "Historical urgent care record documents the 2019 episode."),

    # Q020 - 2021 collision
    ("Q020", 6, 3, "Primary care note records the patient's reported 2021 collision."),

    # Q021 - Postoperative medications
    ("Q021", 19, 3, "Pharmacy dispensing history lists medications dispensed on September 10."),

    # Q022 - Return to work
    ("Q022", 21, 3, "Employer record contains the work-status chronology."),
    ("Q022", 17, 2, "Postoperative note records release to modified duty."),
    ("Q022", 5, 1, "ED discharge record gives the initial work restriction."),

    # Q023 - Total charges
    ("Q023", 20, 3, "Billing summary directly lists total charges."),

    # Q024 - Missing records
    ("Q024", 22, 3, "Records custodian lists all four requested but unproduced items."),

    # Q025 - Missing cervical MRI report
    ("Q025", 22, 3, "Custodian confirms the cervical MRI report was not produced."),
    ("Q025", 9, 1, "Orthopedic consultation gives only a brief reference to the MRI."),

    # Q026 - Missing June 11 injection documentation
    ("Q026", 22, 3, "Custodian confirms the clinical documentation was not located."),
    ("Q026", 20, 1, "Billing summary records a June 11 injection charge, not clinical details."),

    # Q027 - Missing PT visits
    ("Q027", 22, 3, "Custodian confirms PT notes for visits 15 through 22 were not produced."),

    # Q028 - Illegible handwriting
    ("Q028", 13, 3, "Transcription explicitly marks portions of the note as illegible."),
    ("Q028", 22, 2, "Custodian confirms that a legible duplicate was not produced.")
]


# --------------------------------------------------
# SAVE CSV FILES
# --------------------------------------------------

def save_csv(file_path, fieldnames, rows):

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def create_evaluation_dataset():

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    question_columns = [
        "question_id",
        "question",
        "expected_answer",
        "answerable",
        "question_type"
    ]

    relevance_columns = [
        "question_id",
        "page_number",
        "relevance_grade",
        "evidence_note"
    ]

    save_csv(
        QUESTIONS_PATH,
        question_columns,
        evaluation_questions
    )

    relevance_rows = [
        {
            "question_id": question_id,
            "page_number": page_number,
            "relevance_grade": grade,
            "evidence_note": note
        }
        for question_id, page_number, grade, note
        in relevance_labels
    ]

    save_csv(
        RELEVANCE_PATH,
        relevance_columns,
        relevance_rows
    )

    print("Evaluation dataset created successfully!")

    print(f"Total questions: {len(evaluation_questions)}")
    print(f"Total relevance labels: {len(relevance_rows)}")

    print(f"Questions saved to: {QUESTIONS_PATH}")
    print(f"Relevance labels saved to: {RELEVANCE_PATH}")


if __name__ == "__main__":

    create_evaluation_dataset()
