"""Plain-language explanations shown in expanders, written for readers with no data-science background."""
import streamlit as st


def metrics_glossary():
    st.markdown(
        """
**Example for the best model:** the test set holds 20 cataract images and 60 normal images. The model found
19 of the 20 cataracts, missed 1, cleared 51 of the 60 normal images and raised a false alarm on 9.

| Term | Plain meaning | Example |
|---|---|---|
| **Missed cataract** | A cataract image that the model called normal. This is the costly mistake in screening. | 1 of 20 |
| **False alarm** | A normal image that the model called cataract. The person gets an extra eye check they did not need. | 9 of 60 |
| **Accuracy** | Share of all images the model got right. It can look good even when the model ignores cataract. | 70 of 80 = 87.5% |
| **Recall** | Share of real cataracts the model found. High recall means few missed cataracts. | 19 of 20 = 95% |
| **Specificity** | Share of normal images the model correctly cleared. High specificity means few false alarms. | 51 of 60 = 85% |
| **F1-score** | One number that balances recall and precision (the share of flagged images that are truly cataract). | 0.792 |
| **MCC** | Matthews correlation coefficient. It uses all four counts (found, missed, cleared, false alarm) and runs from -1 to +1. Zero means no better than guessing, +1 means perfect. It stays honest when one class is rare. | 0.726 |
        """
    )


def clarity_glossary():
    st.markdown(
        """
Each photograph is converted to grey and measured. A cloudy lens makes the photograph softer, so these values
tend to be **lower for cataract**.

- **Edge density:** the share of pixels that sit on a sharp edge such as a blood vessel boundary. A sharp photograph
  has many edges.
- **Corner density:** the share of pixels that are corner points where lines meet or turn. Fine vessel detail creates
  corners, and blur removes them.
- **Laplacian variance:** the Laplacian filter highlights places where brightness changes quickly. Its variance is a
  standard sharpness score: high means crisp detail, low means blurry.

These are tendencies, not rules. The two classes overlap, so a single measurement cannot decide. The models combine
401 features, of which these three are only a part.
        """
    )


def strategies_glossary():
    st.markdown(
        """
Cataract images are only one in four, so a model can get a good score by mostly saying "normal". Each strategy
tries to correct that.

| Strategy | What it does | Effect on results |
|---|---|---|
| **Baseline** | Trains on the data as it is, with no correction. | Favours the common class. Recall is often lower. |
| **Class weights** | Makes a mistake on a cataract image count about three times as much during training. No new data is created. | Usually raises recall. Can raise false alarms. |
| **ROS** (random oversampling) | Copies randomly chosen cataract training samples until both classes are the same size. | Simple and often effective. The model may memorise the copies. |
| **SMOTE** | Creates new, synthetic cataract samples by blending neighbouring real ones. Works on numeric features, so only the classical models use it. | Gives variety that plain copies lack. Synthetic samples can be unrealistic. |
| **Augmentation** | Creates modified copies of cataract images (flips, rotations, brightness changes) for the neural networks. | Helps when done well. Done badly, the small CNN collapsed to "all normal". |
        """
    )
