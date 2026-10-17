
"""Kestrel Home — Service Request Routing UI."""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Kestrel Home | Service Routing",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Kestrel Home")
st.subheader("AI Service-Request Routing")
st.caption(
    "Enter a customer request to receive a suggested team, "
    "explanation, and review recommendation."
)

with st.sidebar:
    st.header("Service status")
    st.write("API endpoint:", API_URL)

    try:
        health = requests.get(
            f"{API_URL}/health",
            timeout=3,
        )
        if health.ok:
            st.success("Routing API is online")
        else:
            st.warning(f"API returned HTTP {health.status_code}")
    except requests.RequestException:
        st.error("API is unavailable. Start the FastAPI server first.")

st.divider()

with st.form("routing_form"):
    st.subheader("Customer request")

    request_id = st.text_input(
        "Request ID (optional)",
        placeholder="e.g. DEMO-001",
    )

    col1, col2 = st.columns(2)

    with col1:
        channel = st.selectbox(
            "Contact channel",
            ["whatsapp", "chat", "email", "ivr"],
        )

        product_family = st.selectbox(
            "Product",
            [
                "Air Fryer",
                "Mixer Grinder",
                "Water Purifier",
                "Robot Vacuum",
                "Induction Cooktop",
                "Ceiling Fan",
                "Room Heater",
            ],
        )

    with col2:
        warranty_status = st.selectbox(
            "Warranty / plan status",
            [
                "in_warranty",
                "out_of_warranty",
                "shield",
            ],
        )

        source = st.selectbox(
            "Request source",
            ["crm", "legacy_zoho"],
        )

    request_text = st.text_area(
        "Request description",
        placeholder=(
            "Example: My water purifier has stopped working "
            "and needs repair."
        ),
        height=130,
    )

    submitted = st.form_submit_button(
        "Route service request",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not request_text.strip():
        st.error("Please enter the customer's request description.")
    else:
        payload = {
            "request_id": request_id.strip() or None,
            "channel": channel,
            "product_family": product_family,
            "warranty_status": warranty_status,
            "request_text": request_text.strip(),
            "source": source,
        }

        try:
            with st.spinner("Analyzing the request..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=30,
                )

            if response.status_code == 200:
                result = response.json()

                st.divider()
                st.subheader("Routing recommendation")

                st.success(
                    f"Suggested team: {result['predicted_team']}"
                )

                metric1, metric2 = st.columns(2)

                metric1.metric(
                    "Model confidence",
                    f"{result['confidence']:.1%}",
                )

                metric2.metric(
                    "Validation accuracy",
                    f"{result['model_validation_accuracy']:.2%}",
                )

                st.caption(
                    f"Model: {result['model_name']}"
                )

                if result["review_required"]:
                    st.warning(
                        "Human review recommended. "
                        "Do not automatically finalize this routing decision."
                    )
                    for reason in result["review_reasons"]:
                        st.write(f"- {reason}")
                else:
                    st.info(
                        "No review flag was triggered by the current "
                        "provisional confidence rules. This does not "
                        "guarantee that the prediction is correct."
                    )

                st.markdown("#### Why was this team suggested?")
                for reason in result["reasons"]:
                    st.write(f"- {reason}")

                st.caption(result["disclaimer"])

                with st.expander("View full API response"):
                    st.json(result)

            else:
                st.error(
                    f"Routing failed with HTTP {response.status_code}."
                )
                try:
                    st.json(response.json())
                except ValueError:
                    st.text(response.text[:1000])

        except requests.RequestException:
            st.error(
                "Could not reach the routing API. "
                "Check that FastAPI is running at "
                "http://127.0.0.1:8000 and try again."
            )

st.divider()
st.caption(
    "Kestrel Home AI Routing | Local prototype | "
    "Predictions support employee decisions and require operational validation."
)
