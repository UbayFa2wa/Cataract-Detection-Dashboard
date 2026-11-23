"""Concise cataract education: matching info cards and a two-column video/demo panel."""
from pathlib import Path

import cv2
import streamlit as st

from components.data import N_NORMAL, N_CATARACT
from components.fundus_utils import read_image, preprocess, clarity_maps
from components.home_lottie import lottie_doctor
from components.ui import card, goto, page_header

IMG = Path(__file__).resolve().parents[1] / 'assets' / 'img'


def render_what_is_cataract():
    page_header('DESCRIPTIVE ANALYTICS', 'What is <em>cataract</em>?',
                'A cloudy eye lens may reduce the clarity of a fundus photograph.', lottie_doctor, 'doctor_educational')

    st.markdown(
        f'''<div class="edu-grid">
          <div class="edu-card"><h4>👁️ Common symptoms</h4>
            <p>Blurred vision · glare or halos · faded colours · difficulty seeing at night.</p></div>
          <div class="edu-card"><h4>🗂️ What this dataset contains</h4>
            <p><b>{N_NORMAL} Normal</b> and <b>{N_CATARACT} Cataract</b> fundus images.
            Glaucoma and other retinal conditions are outside this two-class study.</p></div>
        </div>''', unsafe_allow_html=True,
    )

    with card('Original dataset examples', 'Representative images; not an independent external test set'):
        cols = st.columns(4)
        samples = [('normal_1.jpg', 'Normal 1'), ('normal_2.jpg', 'Normal 2'),
                   ('cataract_1.jpg', 'Cataract 1'), ('cataract_2.jpg', 'Cataract 2')]
        for col, (filename, caption) in zip(cols, samples):
            col.image(str(IMG / filename), caption=caption, use_container_width=True)

    # Place the original video and original blur/edge experiment together.
    video_col, demo_col = st.columns(2)
    with video_col:
        with card('Short educational video', 'Educational reference', height=540):
            st.video('https://www.youtube.com/watch?v=BXEfJNdXlYM')
            st.caption('Educational reference · not clinical validation.')
    with demo_col:
        with card('Interactive clarity demo', 'Blur a normal fundus image and watch the clarity measures', height=540):
            strength = st.slider('Additional Gaussian blur', 0, 10, 0,
                                 key='blur_demo', help='Educational blur simulation only')
            rgb = read_image((IMG / 'normal_1.jpg').read_bytes())
            enhanced = preprocess(rgb)
            if strength:
                kernel = int(strength * 4) | 1
                enhanced = cv2.GaussianBlur(enhanced, (kernel, kernel), strength)
            edges, _corners, clarity = clarity_maps(enhanced)
            left, right = st.columns(2, gap='small')
            left.image(enhanced, caption='Simulated blur', use_container_width=True)
            right.image(edges, caption='Detected edges', use_container_width=True, clamp=True)
            st.markdown(
                f'<div class="clarity-values">'
                f'<span><b>Edge density</b> {clarity["edge_density"]:.4f}</span>'
                f'<span><b>Laplacian variance</b> {clarity["laplacian_variance"]:.0f}</span>'
                f'</div>', unsafe_allow_html=True,
            )
            st.caption('Simulated blur is not cataract pathology.')

    st.info('Research demonstration only. Diagnosis requires an eye-care professional.')
    if st.button('NEXT: UNDER THE LENS ➜', key='to_lens'):
        goto('go_to_under_the_lens')
