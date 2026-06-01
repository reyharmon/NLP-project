"""
ToS & Privacy Policy Simplifier
Streamlit Web Application

Project: Risk Clause Analysis and Summarization of Terms of Service
Course: COMP6885001 - NLP | Bina Nusantara University
"""

import streamlit as st
import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="ToS Simplifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "ToS & Privacy Policy Simplifier | NLP Final Project | Bina Nusantara University 2025",
    }
)

# ── Risk Keyword Dictionary ──────────────────────────────────
RISK_CATEGORIES = {
    "🔴 Data Collection & Selling": {
        "keywords": ["collect your data", "sell your data", "sell your information",
                     "personal information", "personal data", "data collection"],
        "explanation": "This service collects or may sell your personal data.",
        "severity": "high",
    },
    "🟠 Third Party Sharing": {
        "keywords": ["third party", "third-party", "partners", "affiliates",
                     "share your information", "disclose", "advertisers"],
        "explanation": "Your data may be shared with third parties or business partners.",
        "severity": "high",
    },
    "🟠 Tracking & Cookies": {
        "keywords": ["cookie", "web beacon", "pixel", "fingerprint",
                     "tracking", "monitor your activity", "behavioral"],
        "explanation": "This service tracks your browsing activity and behavior.",
        "severity": "medium",
    },
    "🔴 Binding Arbitration": {
        "keywords": ["binding arbitration", "class action waiver", "class action",
                     "dispute resolution", "waive your right"],
        "explanation": "You waive your right to participate in class action lawsuits.",
        "severity": "high",
    },
    "🟡 Account Termination": {
        "keywords": ["terminate", "suspend your account", "cancel your account",
                     "delete your account", "ban", "at any time without notice"],
        "explanation": "The service can close your account at any time without notice.",
        "severity": "medium",
    },
    "🟡 Location Data": {
        "keywords": ["location", "gps", "geolocation", "precise location",
                     "whereabouts", "geographic"],
        "explanation": "This service collects your physical location data.",
        "severity": "medium",
    },
    "🟡 Data Retention": {
        "keywords": ["retain", "indefinitely", "we keep", "data retention",
                     "deletion request", "after termination"],
        "explanation": "Your data may be stored for an extended or indefinite period.",
        "severity": "medium",
    },
    "🟢 Right to Modify": {
        "keywords": ["we may change", "we reserve the right", "modify these terms",
                     "update this policy", "without notice"],
        "explanation": "The service can change its terms at any time.",
        "severity": "low",
    },
}

# ── Model Loading ────────────────────────────────────────────

HF_MODEL_ID = "reyharmon/t5-base-ToS-corpus"

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_ID, torch_dtype=torch.float32)
    return tokenizer, model, "T5-base Fine-tuned on ToS corpus"


def chunk_text(text: str, max_words: int = 700) -> list:
    words = text.split()
    return [
        ' '.join(words[i: i + max_words])
        for i in range(0, len(words), max_words)
        if words[i: i + max_words]
    ]


def generate_summary(text: str, tokenizer, model) -> str:
    device = next(model.parameters()).device
    word_count = len(text.split())
    chunks = chunk_text(text, max_words=400) if word_count > 400 else [text]
    chunks = chunks[:4]

    summaries = []
    per_chunk_len = max(80, 256 // len(chunks))

    for chunk in chunks:
        # T5 requires task prefix
        prefixed = "summarize: " + chunk
        inputs = tokenizer(
            prefixed,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=per_chunk_len,
                min_length=20,
                num_beams=4,
                length_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        summary = tokenizer.decode(ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)


def detect_risks(text: str) -> dict:
    text_lower = text.lower()
    detected = {}
    for category, info in RISK_CATEGORIES.items():
        found = [kw for kw in info["keywords"] if kw in text_lower]
        if found:
            detected[category] = {
                "found_keywords": found[:3],
                "explanation": info["explanation"],
                "severity": info["severity"],
            }
    return detected


def get_risk_level(risks: dict) -> tuple:
    severities = [v["severity"] for v in risks.values()]
    high_count = severities.count("high")
    medium_count = severities.count("medium")

    if high_count >= 3:
        return "VERY HIGH", "error"
    elif high_count >= 1 or medium_count >= 3:
        return "HIGH", "error"
    elif medium_count >= 1:
        return "MEDIUM", "warning"
    elif risks:
        return "LOW", "info"
    else:
        return "SAFE", "success"


# ── CSS ─────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    /* Light mode: dark text */
    [data-theme="light"] .main-title { color: #1f2937; }
    /* Dark mode: white text */
    [data-theme="dark"]  .main-title { color: #ffffff; }
    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    /* Hide deploy button */
    [data-testid="stDeployButton"] { display: none !important; }
    /* Hide running man animation, keep "Running" text */
    [data-testid="stStatusWidget"] svg { display: none !important; }
    [data-testid="stStatusWidget"] img { display: none !important; }
    .summary-box {
        background-color: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        font-size: 1rem;
        line-height: 1.7;
        color: #1f2937 !important;
        font-family: sans-serif;
    }
    .summary-box p {
        color: #1f2937 !important;
        margin: 0;
    }
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Main App ─────────────────────────────────────────────────
def main():
    inject_css()

    # Header
    st.markdown('<p class="main-title">📄 ToS & Privacy Policy Simplifier</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Paste your Terms of Service or Privacy Policy text — '
        'we will summarize it and highlight risky clauses for you.</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Input")
        text_input = st.text_area(
            label="Input",
            height=380,
            placeholder=(
                "Paste your Terms of Service, Privacy Policy, or Cookie Policy here...\n\n"
                "Example:\n"
                "By using our Service, you agree that we may collect your personal data "
                "including name, email, location, and browsing history. We may share this "
                "information with third-party advertising partners. You agree to binding "
                "arbitration and waive your right to class action lawsuits..."
            ),
            label_visibility="collapsed",
        )

        word_count = len(text_input.split()) if text_input.strip() else 0
        st.caption(f"Word count: **{word_count:,}**")

        analyze_btn = st.button(
            "Analyze & Summarize",
            type="primary",
            use_container_width=True,
            disabled=(word_count < 30),
        )

        if 0 < word_count < 30:
            st.caption("Please enter at least 30 words to analyze.")

    with right:
        st.subheader("Results")

        if analyze_btn and word_count >= 30:
            with st.spinner("Loading model... (first time only)"):
                tokenizer, model, model_info = load_model()

            st.caption(f"Model: `{model_info}`")

            with st.spinner("Analyzing document..."):
                summary = generate_summary(text_input, tokenizer, model)
                risks = detect_risks(text_input)
                risk_level, alert_type = get_risk_level(risks)

            # Summary
            st.markdown("#### Summary")
            st.markdown(
                f'<div class="summary-box"><p>{summary}</p></div>',
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # Risk Level
            st.markdown("#### Overall Risk Level")
            if alert_type == "error":
                st.error(f"Risk Level: **{risk_level}** — {len(risks)} risk categories detected")
            elif alert_type == "warning":
                st.warning(f"Risk Level: **{risk_level}** — {len(risks)} risk categories detected")
            elif alert_type == "info":
                st.info(f"Risk Level: **{risk_level}** — {len(risks)} risk categories detected")
            else:
                st.success("Risk Level: **SAFE** — No major risk clauses found")

            # Risk Categories
            if risks:
                st.markdown("#### Detected Risk Clauses")
                for category, info in risks.items():
                    with st.expander(category):
                        st.markdown(f"**Explanation:** {info['explanation']}")
                        st.markdown(
                            f"**Keywords found:** `{'`, `'.join(info['found_keywords'])}`"
                        )

        elif not analyze_btn:
            st.markdown("""
            #### How to Use

            1. **Paste** your ToS / Privacy Policy text on the left
            2. Click **Analyze & Summarize**
            3. Get a plain-English summary + risk analysis

            ---

            #### What We Detect

            | Category | Description |
            |----------|-------------|
            | 🔴 Data Collection | Your data is collected or sold |
            | 🟠 Third Party Sharing | Data shared with partners |
            | 🟠 Tracking & Cookies | Browsing activity tracked |
            | 🔴 Binding Arbitration | Your legal rights are limited |
            | 🟡 Account Termination | Account can be closed anytime |
            | 🟡 Location Data | GPS location is collected |
            | 🟡 Data Retention | Data stored indefinitely |
            | 🟢 Right to Modify | Terms can change without notice |

            ---

            #### Supported Document Types
            - Terms of Service (ToS)
            - Privacy Policy
            - Cookie Policy
            - End User License Agreement (EULA)
            """)

    # Footer
    st.markdown(
        '<div class="footer">NLP Final Project · COMP6885001 · Bina Nusantara University · 2026<br>'
        'Andrey Apriliady · Ezra Mayurga · Keanu Stadeva</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
