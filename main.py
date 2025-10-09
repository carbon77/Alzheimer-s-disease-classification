import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

from model import transform, load_model

# Page configuration
st.set_page_config(
    page_title="Классификация болезни Альцгеймера",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_prediction_bar_chart(predictions, class_names):
    df = pd.DataFrame({
        'Class': class_names,
        'Confidence': predictions
    })

    fig = px.bar(df, x='Confidence', y='Class', orientation='h',
                 color='Confidence',
                 color_continuous_scale='Viridis',
                 title='Вероятность предсказания по типу')

    fig.update_layout(height=400, showlegend=False)
    return fig


def main():
    st.title('🧠 Классификация болезни Альцгеймера по данным МРТ')

    st.session_state.model = load_model()
    class_names = {
        0: "Лёгкая форма деменции",
        1: "Умеренная деменция",
        2: "Нет деменции",
        3: "Очень лёгкая форма деменции"
    }

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📁 Загрузите изображение МРТ")

        uploaded_file = st.file_uploader(
            "Выберите изображение МРТ",
            type=[".jpeg", ".jpg", ".png"],
            key="input_image",
            label_visibility="collapsed"
        )

        run_model_btn = st.button(
            "🚀 Проанализировать изображение",
            disabled=st.session_state.input_image is None,
            use_container_width=True
        )

        if uploaded_file is not None:
            st.image(uploaded_file, "Загруженное изображение", use_container_width=True)

    with col2:
        st.subheader("📊 Результаты анализа")

        if run_model_btn and uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            img_col, result_col = st.columns([1, 2])

            with img_col:
                st.image(image, caption="Изображение МРТ для анализа", use_container_width=True)

            with st.spinner("🔬 Выполняется анализ..."):
                image_tensor = transform(image)
                logits = st.session_state.model(image_tensor)

            softmax = nn.Softmax()
            predictions = softmax(logits) * 100
            predicted_class = torch.argmax(logits).item()

            with result_col:
                st.write("### 🎯 Диагноз")
                st.write(f"**Предсказанный тип:** {class_names.get(predicted_class, f'Class {predicted_class}')}")
                st.write(f"**Уверенность:** {predictions[predicted_class]:.2f}%")

                st.markdown("### 📈 Детальный анализ")

                prediction_data = []
                for i, confidence in enumerate(predictions):
                    prediction_data.append({
                        'Class': class_names.get(i, f'Class {i + 1}'),
                        'Confidence': confidence.item(),
                        'IsPredicted': i == predicted_class
                    })

                for pred in prediction_data:
                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        st.write(f"**{pred['Class']}** {'🎯' if pred['IsPredicted'] else ''}")
                    with col_b:
                        st.write(f"{pred['Confidence']:.2f}%")

                st.plotly_chart(create_prediction_bar_chart(
                    [p['Confidence'] for p in prediction_data],
                    [p['Class'] for p in prediction_data]
                ), use_container_width=True)

        else:
            st.info("Загрузите изображение МРТ, чтобы увидеть результаты")


if __name__ == "__main__":
    main()
