import os
import pickle
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "data")

with open(os.path.join(data_dir, "shap_output.pkl"), "rb") as f:
    shap_output = pickle.load(f)

with open(os.path.join(data_dir, "encoders.pkl"), "rb") as f:
    encoders = pickle.load(f)

feature_cols = shap_output["features"]
shap_values = shap_output["values"]
shap_data = shap_output["data"]

# Convert to DataFrame
shap_df = pd.DataFrame(shap_values, columns=feature_cols)
data_df = pd.DataFrame(shap_data, columns=feature_cols)

role_cols = [
    "blue_top", "blue_jng", "blue_mid", "blue_bot", "blue_sup",
    "red_top", "red_jng", "red_mid", "red_bot", "red_sup"
]

# Let's decode the champion names
decoded_data = data_df.copy()
for col in role_cols:
    enc_col = col + "_enc"
    le = encoders[col]
    # For each row, decode the integer back to name
    decoded_data[col] = decoded_data[enc_col].apply(lambda x: le.classes_[int(x)])

# We want to get the SHAP contribution for each champion.
# For each row and each role column:
# if a champion is present in that role, their SHAP contribution in that role is shap_df[role_col + "_enc"]
# Let's aggregate for all champions:
champ_contributions = {}

for i in range(len(data_df)):
    for col in role_cols:
        enc_col = col + "_enc"
        champ_name = decoded_data.loc[i, col]
        shap_val = shap_df.loc[i, enc_col]
        # Since a positive SHAP value for red team means it contributed negatively to blue's win probability,
        # wait! Let's think: is the SHAP value relative to blue winning?
        # Yes, target is blue_win!
        # So a positive SHAP value for a Blue champion means it increases Blue's win chance.
        # A positive SHAP value for a Red champion means it increases Blue's win chance? No, wait!
        # If a Red champion has a negative SHAP value, it means it decreased Blue's win chance (which is good for Red!).
        # Let's just calculate the mean absolute SHAP value or the actual SHAP value.
        # The PDF says: "To get which specific champions are OP, group by champion name and average their SHAP contribution across all matches they appeared in."
        # Wait, if they are on Blue team, the SHAP value represents their contribution to blue winning.
        # If they are on Red team, the SHAP value represents their contribution to blue winning (so positive means they helped Blue, negative means they helped Red).
        # To make it consistent, the contribution of a champion to their OWN team's win rate:
        # For Blue champion: +shap_val
        # For Red champion: -shap_val
        # Let's see:
        contribution = shap_val if col.startswith("blue") else -shap_val
        
        if champ_name not in champ_contributions:
            champ_contributions[champ_name] = []
        champ_contributions[champ_name].append(contribution)

champ_op_metrics = []
for champ, contribs in champ_contributions.items():
    mean_contrib = np.mean(contribs)
    abs_mean_contrib = np.mean(np.abs(contribs))
    count = len(contribs)
    champ_op_metrics.append({
        "champion": champ,
        "mean_contribution": mean_contrib,
        "mean_abs_contribution": abs_mean_contrib,
        "count": count
    })

op_df = pd.DataFrame(champ_op_metrics)
# Sort by mean contribution to see who consistently pushes win probability up!
print("\nTOP 10 CHAMPIONS BY MEAN SHAP CONTRIBUTION:")
print(op_df.sort_values(by="mean_contribution", ascending=False).head(10))

print("\nTOP 10 CHAMPIONS BY MEAN ABSOLUTE SHAP CONTRIBUTION:")
print(op_df.sort_values(by="mean_abs_contribution", ascending=False).head(10))
