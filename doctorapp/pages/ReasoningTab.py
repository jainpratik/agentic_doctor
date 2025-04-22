import streamlit as st
from tools.langgraph_neo4j import run_langgraph_pipeline
from tools.reasoning_agent import run_reasoning_pipeline


def reasoning_ui():
    st.markdown("## 🧠 Agentic Reasoning Engine")

    notes = st.text_area("📝 Paste clinical notes or user query", height=150)
    symptom_input = st.text_input("💬 List known symptoms (comma-separated)")

    if st.button("Run Agentic Reasoning"):
        symptoms = [s.strip() for s in symptom_input.split(",") if s.strip()]
        with st.spinner("Running full diagnostic pipeline..."):
            hypothesis, structured = run_reasoning_pipeline(notes, symptoms)

        # Display step-by-step results
        st.markdown("### 🧠 Step 1: Initial Hypothesis")
        st.info(hypothesis)

        st.markdown("### 🔍 Step 2: Knowledge Graph Findings")
        st.markdown(f"**Top diagnosis:** `{structured['diagnosis']}`")
        st.markdown(
            f"**Supported by symptoms:** {', '.join(structured['symptom_support'])}"
        )

        st.markdown("### 📋 Step 3: Final Structured Output")
        st.json(structured)

        st.markdown("### 💡 Step 4: Explanation + Next Steps")
        st.markdown(f"- **Tests:** {', '.join(structured['tests'])}")
        st.markdown(f"- **Rule out:** {', '.join(structured['rule_out'])}")
        st.markdown(f"- **Next steps:** {', '.join(structured['next_steps'])}")
        st.markdown(f"- **Follow-ups:** {', '.join(structured['follow_ups'])}")


def langgraph_dag_ui():
    st.markdown("## 🧠 LangGraph Reasoning DAG")

    notes = st.text_area(
        "📥 Enter clinical notes (symptoms comma-separated):", height=150
    )

    if st.button("Run LangGraph DAG"):
        with st.spinner("Running DAG through Neo4j..."):
            final_state = run_langgraph_pipeline(notes)

        st.success("✅ LangGraph DAG completed!")
        st.markdown("### 🔍 Final State:")
        st.json(final_state)
