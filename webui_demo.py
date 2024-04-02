import streamlit as st
from library.page_tools_demo import PageTools
from library.page_info_demo import PageInfo
from library.components import footer_style, footer
import hydralit_components as hc
import streamlit_lottie
import time
import json
import os

st.set_page_config(
    page_title='',
    page_icon="",
    initial_sidebar_state="expanded"
)


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


if 'lottie' not in st.session_state:
    st.session_state.lottie = False

if not st.session_state.lottie:
    lottfinder = load_lottiefile("img/lottielogoAI.json")
    st.lottie(lottfinder, speed=1.3, loop=False)
    time.sleep(2)
    st.session_state.lottie = True
    st.rerun()

max_width_str = f"max-width: {75}%;"

st.markdown(f"""
        <style>
        .appview-container .main .block-container{{{max_width_str}}}
        </style>
        """,
            unsafe_allow_html=True,
            )

st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;

                }
        </style>
        """, unsafe_allow_html=True)

# Footer

st.markdown(footer_style, unsafe_allow_html=True)

INFO = 'Информация'
TOOLS = 'Инструменты'

tabs = [
    INFO,
    TOOLS
]

option_data = [
    {'icon': "📝", 'label': INFO},
    {'icon': "📐", 'label': TOOLS}
]

chosen_tab = hc.option_bar(
    option_definition=option_data,
    title='',
    key='PrimaryOptionx',
    horizontal_orientation=True)

if chosen_tab == INFO:
    info_page = PageInfo()
    info_page.run()
elif chosen_tab == TOOLS:
    segmentation_page = PageTools()
    segmentation_page.run()

for i in range(4):
    st.markdown('#')
st.markdown(footer, unsafe_allow_html=True)

# Credit
st.sidebar.image("img/logoAI.png")

st.sidebar.title("Обращения")
st.sidebar.markdown(
    "[Нужна помощь? 🆘](https://github.com/deneal123/MuseumAI/issues/new?assignees=&labels=help+wanted&projects=&template=help.md&title=%5BHELP%5D)")
st.sidebar.markdown(
    "[Есть предложения? 💡](https://github.com/deneal123/MuseumAI/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.md&title=%5BFEATURE%5D)")
st.sidebar.markdown("[Хотите пообщаться? 🙋🏼‍♂](https://github.com/deneal123/MuseumAI/discussions)")
