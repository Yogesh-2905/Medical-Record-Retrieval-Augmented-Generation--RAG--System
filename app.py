import streamlit as st
from generate_answer import generate_answer


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Medical Record RAG System",
    page_icon="🏥",
    layout="wide"
)


# --------------------------------------------------
# 2. Application Header
# --------------------------------------------------

st.title("🏥 Medical Record RAG System")

st.markdown(
    """
    ### AI-Powered Medical Record Question Answering

    Ask questions about the available medical records.
    The system retrieves relevant information and generates
    answers grounded in the retrieved context.
    """
)

st.divider()


# --------------------------------------------------
# 3. Sidebar - System Information
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ System Configuration")

    st.success("System Configuration")

    st.write("**Embedding Model:**")
    st.write("BAAI/bge-small-en-v1.5")

    st.write("**Embedding Dimensions:**")
    st.write("384")

    st.write("**Vector Database:**")
    st.write("ChromaDB")

    st.write("**Collection:**")
    st.write("medical_records")

    st.write("**Similarity Metric:**")
    st.write("Cosine Similarity")

    st.write("**Top-K:**")
    st.write("8")

    st.write("**Reranking:**")
    st.write("OFF")

    st.divider()

    st.caption(
        "Medical Record RAG System | Demo Application"
    )


# --------------------------------------------------
# 4. User Question Input
# --------------------------------------------------

st.subheader("💬 Ask Your Question")

with st.form("medical_question_form"):

    user_question = st.text_area(
        "Enter your medical record question:",
        placeholder="Type your question here...",
        height=120
    )

    submit_button = st.form_submit_button(
        "🔍 Generate Answer",
        use_container_width=True
    )


# --------------------------------------------------
# 5. Generate Answer
# --------------------------------------------------

if submit_button:

    if not user_question.strip():

        st.warning("Please enter a question before submitting.")

    else:

        st.divider()

        st.subheader("📝 Generated Answer")

        try:

            with st.spinner(
                "Retrieving relevant medical records and generating answer..."
            ):

                answer = generate_answer(user_question.strip())

            if not answer:
                st.warning("The system returned an empty response.")

            elif answer.strip() == "I don't know.":

                st.warning("I don't know.")

                st.caption(
                    "The available retrieved context was insufficient "
                    "to answer this question."
                )

            else:

                st.success("Answer generated successfully.")

                st.markdown(answer)

        except Exception as e:

            st.error(
                "An error occurred while processing your question."
            )

            st.exception(e)


# --------------------------------------------------
# 6. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "This application generates answers based on retrieved "
    "medical record context. Always verify important information "
    "against the original medical records."
)
