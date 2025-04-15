import streamlit as st

st.title("Сервис для улучшения качества изображения")
st.markdown(
    "Данный сервис позволяет повышать разрешение изображения, убирать смазы и шумы с него. Ссылка на проект: [https://github.com/vladparh/image_enhancement.git](https://github.com/vladparh/image_enhancement.git)"
)
st.subheader("Повышения разрешения")
st.markdown(
    "Данная операция позволяет повысить разрешение, при этом повышается детализация, убираются шумы и артефакты сжатия."
)
st.image(
    "src/services/streamlit/examples/ex_1.png", caption="Пример повышения разрешения"
)
st.subheader("Удаление смазов")
st.markdown(
    "Данная операция позволяет удалять смазы с изображения, возникающие в следствии тряски камеры. Лучше всего работает для изображений размером порядка 1 Мпикс."
)
st.image("src/services/streamlit/examples/ex_2.png", caption="Пример удаления смазов")
st.subheader("Удаление шумов")
st.markdown(
    "Данная операция позволяет удалять шумы с изображений, возникающие из-за особенностей устройства CMOS-матриц, которые используются в современных смартфонах и фотокамерах."
)
st.image("src/services/streamlit/examples/ex_3.png", caption="Пример удаления шумов")
