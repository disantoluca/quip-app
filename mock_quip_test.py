import streamlit as st

# Manually inject one mock doc into the session
if "docs" not in st.session_state:
    st.session_state.docs = []

st.session_state.docs.append({
    "source": "mock:montenegro",
    "text": """Driving rules in Montenegro: 
    U-turns are generally prohibited in areas marked with solid lines 
    or near intersections. 
    However, they are permitted when visibility and traffic conditions allow, 
    unless a sign explicitly forbids it. 
    Always yield to oncoming traffic before making a U-turn.""",
    "meta": {
        "id": "mock001",
        "title": "Montenegro Driving Rules",
        "link": "https://example.com/montenegro-driving",
        "created_usec": "20250101000000",
        "updated_usec": "20250102000000"
    }
})

st.success("✅ Mock document added. Go to the 'Ask Questions' tab to test FAISS.")