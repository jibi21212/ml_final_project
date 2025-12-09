import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------------
# Model definition (same as notebook)
# -------------------------------
class OutcomeNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 2)  # 2 classes: A loses (0), A wins (1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits

# -------------------------------
# Load model + data once
# -------------------------------
@st.cache_resource
def load_model_and_data():
    # Load fighter embeddings
    fighter_emb_df = pd.read_csv("fighter_embeddings.csv")

    # Load model bundle
    bundle = torch.load("outcome_model_bundle.pt", map_location="cpu")
    mean_match = bundle["mean_match"]          # tensor (1, 96)
    std_match = bundle["std_match"]            # tensor (1, 96)

    input_dim = mean_match.shape[1]            # should be 96
    model = OutcomeNet(input_dim)
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    # Build lookup tables
    emb_cols = [c for c in fighter_emb_df.columns if c.startswith("emb_")]
    emb_by_id = fighter_emb_df.set_index("fighters_id")[emb_cols]

    # Map name -> some fighters_id (first occurrence)
    name_to_id = (
        fighter_emb_df.groupby("fighter_name")["fighters_id"]
        .first()
        .to_dict()
    )

    all_names = sorted(fighter_emb_df["fighter_name"].unique().tolist())

    return model, mean_match, std_match, emb_by_id, name_to_id, all_names

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("UFC Fight Outcome Predictor")
st.write(
    "Select two fighters to predict the probability that Fighter A wins "
    "based on learned style embeddings."
)

model, mean_match, std_match, emb_by_id, name_to_id, all_names = load_model_and_data()

col1, col2 = st.columns(2)

with col1:
    fighterA_name = st.selectbox("Fighter A", all_names, index=0)
with col2:
    fighterB_name = st.selectbox("Fighter B", all_names, index=1)

if st.button("Predict"):
    if fighterA_name == fighterB_name:
        st.error("Please choose two different fighters.")
    else:
        # Get ids
        fidA = name_to_id.get(fighterA_name, None)
        fidB = name_to_id.get(fighterB_name, None)

        if fidA not in emb_by_id.index or fidB not in emb_by_id.index:
            st.error("One of the fighters is missing an embedding.")
        else:
            # Build feature vector [A, B, |A-B|]
            embA = torch.tensor(emb_by_id.loc[fidA].values, dtype=torch.float32).unsqueeze(0)
            embB = torch.tensor(emb_by_id.loc[fidB].values, dtype=torch.float32).unsqueeze(0)
            x = torch.cat([embA, embB, torch.abs(embA - embB)], dim=1)

            # Standardize using training mean/std
            x_scaled = (x - mean_match) / std_match

            with torch.no_grad():
                logits = model(x_scaled)
                probs = torch.softmax(logits, dim=1).numpy()[0]

            prob_A_wins = probs[1]
            prob_B_wins = probs[0]

            st.subheader("Prediction")
            st.write(f"**Probability {fighterA_name} wins:** {prob_A_wins * 100:.1f}%")
            st.write(f"**Probability {fighterB_name} wins:** {prob_B_wins * 100:.1f}%")
