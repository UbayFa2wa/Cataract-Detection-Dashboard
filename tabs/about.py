"""About page: scope, method, data source, limits and references."""
import streamlit as st

from components.ui import hero, takeaway

APA = [
    "Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357.",
    "Chicco, D., & Jurman, G. (2023). The Matthews correlation coefficient (MCC) should replace the ROC AUC as the standard metric for assessing binary classification. *BioData Mining, 16*, Article 4.",
    "GBD 2019 Blindness and Vision Impairment Collaborators. (2021). Causes of blindness and vision impairment in 2020 and trends over 30 years. *The Lancet Global Health, 9*(2), e144–e160.",
    "Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE CVPR*, 4510–4520.",
    "Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th ICML*, 6105–6114.",
]


def render_about():
    hero("ABOUT", "About this <em>project</em>",
         "Course project for XBDS3024N Image Processing and Computer Vision, built by Bayu Fatwa Negara.")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### 🎯 Aim")
            st.markdown(
                "Compare six models for cataract versus normal fundus classification under a 3:1 class "
                "imbalance, and judge them with metrics that do not hide a missed minority class."
            )
            st.markdown("#### 🧪 Method")
            st.markdown(
                "- **Data:** 400 colour fundus photographs, 300 normal and 100 cataract, split 80/20 with stratification.\n"
                "- **Preprocessing:** crop the black border, resize to 224 × 224, CLAHE on the lightness channel.\n"
                "- **Classical models:** SVM, Random Forest and Logistic Regression on 401 handcrafted features "
                "(colour histograms, GLCM and LBP texture, HOG, and three clarity measurements) reduced with PCA.\n"
                "- **Deep models:** a small CNN, MobileNetV2 and EfficientNetB0.\n"
                "- **Imbalance strategies:** none, class weights, random oversampling, SMOTE and augmentation.\n"
                "- **Metrics:** recall, specificity, F1, MCC, ROC-AUC and Wilson intervals."
            )
    with c2:
        with st.container(border=True):
            st.markdown("#### ⚠️ Limits and ethics")
            st.markdown(
                "- One public dataset of 400 images. The test set holds only 20 cataract images, so every recall "
                "estimate has a wide interval.\n"
                "- Scores are not calibrated probabilities.\n"
                "- Uploaded images are processed in memory and not stored.\n"
                "- The tool is a **research demonstration, not a medical device**. Only an eye-care professional "
                "can diagnose a cataract."
            )
            st.markdown("#### 🗂️ Data source")
            st.markdown("Fundus images: [yiweichen04/retina_dataset](https://github.com/yiweichen04/retina_dataset).")

    with st.container(border=True):
        st.markdown("#### 📚 References")
        for ref in APA:
            st.markdown(f"- {ref}")
        st.caption("Check each reference against the report's reference list before submission.")

    takeaway("The notebook in the repository holds the full analysis. This dashboard turns its results into a story "
             "from description to prescription.")
