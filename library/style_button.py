import streamlit as st


def _center():
    # Добавляем стили для центрирования кнопки
    st.markdown("""<style>
    
    /* Выравниваем кнопку по центру */
    
    div.stButton > button {
      display: block;
      margin: 0 auto;
    }
    
    </style>""", unsafe_allow_html=True)


def _move_button(left=0, right=0, top=0, bottom=0):
    # Генерируем стили для перемещения кнопки
    style = f"""
    <style>
        /* Перемещаем кнопку */
        div.stButton > button {{
            position: relative;
            left: {left}px;
            right: {right}px;
            top: {top}px;
            bottom: {bottom}px;
        }}
    </style>
    """
    # Применяем стили
    st.markdown(style, unsafe_allow_html=True)
