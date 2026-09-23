import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
})

# ===== 多语言支持 =====
LANGUAGES = {
    "English": "en",
    "中文": "zh",
}

TEXTS = {
    "en": {
        "title": "🍱 HSR-Derived Front-of-Pack Labelling Prototype",
        "subtitle": "Binary Endorsement-Style Classification",
        "description": (
            "This prototype uses a locked XGBoost model and three routinely "
            "available on-pack nutrients to generate an approximate binary "
            "HSR-derived classification for prepackaged ready foods."
        ),
        "target_audience": "🎯 Intended Users",
        "audience_desc": (
            "Designed for researchers and public-health practitioners exploring "
            "food-supply monitoring in settings where complete HSR inputs are unavailable."
        ),
        "problem_statement": "📊 Problem Statement",
        "problem_desc": (
            "Direct HSR calculation requires nutritional and compositional inputs "
            "that may not be routinely reported on food labels."
        ),
        "solution": "💡 Prototype Approach",
        "solution_desc": (
            "The locked model uses sodium, protein and energy to classify products "
            "into Healthy or Unhealthy HSR-derived categories and provides "
            "SHAP-based explanations of individual predictions."
        ),
        "mission": "🚀 Intended Application",
        "mission_desc": (
            "This prototype demonstrates a reduced-input approach for exploratory "
            "food-supply monitoring and comparisons among products within the same "
            "ready-food category when complete inputs for direct HSR calculation "
            "are unavailable. The Healthy and Unhealthy outputs are predicted labels "
            "defined by the HSR threshold of 3.5 and should not be interpreted as "
            "independent assessments of overall product healthfulness or as formal "
            "Health Star Ratings."
        ),
        "input_variables": "🔢 Available On-Pack Input Variables",
        "protein_label": "Protein (g/100 g)",
        "sodium_label": "Sodium (mg/100 g)",
        "energy_label": "Energy (kJ/100 g)",
        "predict_button": "🧮 Generate Predicted HSR-Derived Category",
        "prediction_result": "🔍 Prediction Result",
        "healthy": "✅ Healthy (predicted HSR-derived label; HSR ≥3.5)",
        "unhealthy": "⚠️ Unhealthy (predicted HSR-derived label; HSR <3.5)",
        "confidence": "Model-Estimated Probability",
        "feature_importance": "📊 Feature Importance",
        "shap_plot": "📊 SHAP Force Plot",
        "base_value": "Baseline Model Output",
        "final_prediction": "Final Model Output",
        "expand_shap": "Click to view the SHAP force plot",
        "shap_success": "✅ SHAP force plot created (Matplotlib version)!",
        "shap_html_success": "✅ SHAP force plot created (HTML backup version)!",
        "shap_custom_success": (
            "✅ SHAP force plot created "
            "(custom version with predictor names)!"
        ),
        "shap_table": "📊 SHAP Values Table",
        "shap_table_info": (
            "💡 SHAP values show how each input contributes to the model output."
        ),
        "positive_impact": (
            "Positive impact (toward the Healthy HSR-derived label)"
        ),
        "negative_impact": (
            "Negative impact (toward the Unhealthy HSR-derived label)"
        ),
        "warning_input": (
            "⚠️ Please enter values for the required input variables before predicting."
        ),
        "input_tip": (
            "💡 Enter the nutrient values exactly as declared per 100 g on the "
            "product package."
        ),
        "model_error": (
            "❌ Prediction cannot proceed because the required model or scaler "
            "files are unavailable."
        ),
        "prediction_failed": "Prediction failed",
        "shap_failed": "SHAP analysis failed",
        "shap_unavailable": (
            "💡 A SHAP explanation is unavailable, but feature importance is "
            "shown above."
        ),
        "footer": (
            "Developed using Streamlit and XGBoost · "
            "For exploratory research use only."
        ),
        "feature_names": ["Sodium", "Protein", "Energy"],
        "chart_feature_names": ["Sodium", "Protein", "Energy"],
    },

    "zh": {
        "title": "🍱 HSR衍生正面标签原型",
        "subtitle": "二分类推荐式标签",
        "description": (
            "本原型使用已锁定的XGBoost模型，根据包装上常规标示的三项营养素，"
            "为预包装即食食品生成近似的二分类HSR衍生结果。"
        ),
        "target_audience": "🎯 预期使用者",
        "audience_desc": (
            "面向在缺少完整HSR计算所需信息的情况下，开展食品供应监测探索的"
            "研究人员和公共卫生实践人员。"
        ),
        "problem_statement": "📊 问题陈述",
        "problem_desc": (
            "直接计算HSR需要多项营养和食品组成信息，而这些信息未必都会在"
            "食品标签上常规标示。"
        ),
        "solution": "💡 原型方法",
        "solution_desc": (
            "已锁定的模型使用钠、蛋白质和能量，将产品划分为“健康”或“不健康”"
            "HSR衍生类别，并使用SHAP解释各输入变量对单次预测的贡献。"
        ),
        "mission": "🚀 预期应用",
        "mission_desc": (
            "当无法获得直接计算HSR所需的完整信息时，本原型展示了一种基于较少"
            "输入的探索性食品供应监测方法，并可辅助比较同一即食食品类别内产品"
            "的相对差异。“健康”和“不健康”仅指以HSR 3.5为界值定义的预测标签，"
            "不代表对产品整体健康程度的独立评价，也不等同于正式的健康星级评分。"
        ),
        "input_variables": "🔢 包装上可获得的输入变量",
        "protein_label": "蛋白质（g/100 g）",
        "sodium_label": "钠（mg/100 g）",
        "energy_label": "能量（kJ/100 g）",
        "predict_button": "🧮 生成预测的HSR衍生类别",
        "prediction_result": "🔍 预测结果",
        "healthy": "✅ 健康（预测的HSR衍生标签；HSR ≥3.5）",
        "unhealthy": "⚠️ 不健康（预测的HSR衍生标签；HSR <3.5）",
        "confidence": "模型估计概率",
        "feature_importance": "📊 特征重要性",
        "shap_plot": "📊 SHAP力图",
        "base_value": "模型基准输出",
        "final_prediction": "最终模型输出",
        "expand_shap": "点击查看SHAP力图",
        "shap_success": "✅ SHAP力图已生成（Matplotlib版本）！",
        "shap_html_success": "✅ SHAP力图已生成（HTML备用版本）！",
        "shap_custom_success": "✅ SHAP力图已生成（包含变量名称的自定义版本）！",
        "shap_table": "📊 SHAP值表格",
        "shap_table_info": "💡 SHAP值表示各输入变量对模型输出的贡献。",
        "positive_impact": "正向贡献（推动预测趋向“健康”HSR衍生标签）",
        "negative_impact": "负向贡献（推动预测趋向“不健康”HSR衍生标签）",
        "warning_input": "⚠️ 请填写所需的输入变量后再进行预测。",
        "input_tip": "💡 请按照产品包装标示填写每100 g的营养素数值。",
        "model_error": "❌ 缺少所需的模型或标准化器文件，无法进行预测。",
        "prediction_failed": "预测失败",
        "shap_failed": "SHAP分析失败",
        "shap_unavailable": (
            "💡 当前无法生成SHAP解释，但上方仍显示特征重要性。"
        ),
        "footer": (
            "使用Streamlit和XGBoost开发 · 仅供探索性研究使用。"
        ),
        "feature_names": ["钠", "蛋白质", "能量"],
        "chart_feature_names": ["Sodium", "Protein", "Energy"],
    },
}

st.set_page_config(
    page_title="Nutritional Quality Classifier",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_language():
    col1, col2, col3 = st.columns([1, 1, 6])
    with col1:
        lang_choice = st.selectbox("🌐 Language", list(LANGUAGES.keys()))
    return TEXTS[LANGUAGES[lang_choice]]


texts = get_language()

st.markdown(f"""
<div style="text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">{texts['title']}</h1>
    <p style="color: #f0f0f0; margin: 0.5rem 0 0 0; font-size: 1.2rem;">{texts['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #28a745; margin-bottom: 2rem;">
    <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{texts['description']}</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="color: #1976d2; margin: 0 0 0.5rem 0;">{texts['target_audience']}</h4>
        <p style="margin: 0; font-size: 0.9rem;">{texts['audience_desc']}</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div style="background: #f3e5f5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="color: #7b1fa2; margin: 0 0 0.5rem 0;">{texts['problem_statement']}</h4>
        <p style="margin: 0; font-size: 0.9rem;">{texts['problem_desc']}</p>
    </div>
    """, unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    st.markdown(f"""
    <div style="background: #e8f5e8; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="color: #2e7d32; margin: 0 0 0.5rem 0;">{texts['solution']}</h4>
        <p style="margin: 0; font-size: 0.9rem;">{texts['solution_desc']}</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div style="background: #fff3e0; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="color: #f57c00; margin: 0 0 0.5rem 0;">{texts['mission']}</h4>
        <p style="margin: 0; font-size: 0.9rem;">{texts['mission_desc']}</p>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        return joblib.load("XGBoost_retrained_model.pkl")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


@st.cache_resource
def load_scaler():
    try:
        return joblib.load("scaler2.pkl")
    except Exception as e:
        st.error(f"Scaler loading failed: {e}")
        return None


@st.cache_resource
def load_background():
    try:
        return np.load("background_data.npy")
    except Exception as e:
        st.error(f"background_data loading failed: {e}")
        return None


@st.cache_resource
def load_feature_names():
    try:
        names = np.load("feature_names.npy", allow_pickle=True)
        return [str(x) for x in names.tolist()]
    except Exception:
        return ["Sodium", "Protein", "Energy"]


model = load_model()
scaler = load_scaler()
background_data = load_background()
chart_names = load_feature_names()

if model is None or scaler is None or background_data is None:
    st.error(texts["model_error"])
    st.stop()

st.sidebar.markdown(f"## {texts['input_variables']}")
st.sidebar.markdown(f"""
<div style="background: #f0f8ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
    <p style="margin: 0; font-size: 0.9rem; color: #1976d2;">
        <strong>{texts['input_tip']}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

sodium = st.sidebar.number_input(texts["sodium_label"], min_value=0.0, step=1.0, help="mg/100 g")
protein = st.sidebar.number_input(texts["protein_label"], min_value=0.0, step=0.1, help="g/100 g")
energy = st.sidebar.number_input(texts["energy_label"], min_value=0.0, step=1.0, help="kJ/100 g, not kcal")

if st.sidebar.button(texts["predict_button"], type="primary", use_container_width=True):
    try:
        input_data = np.array([[sodium, protein, energy]], dtype=float)
        input_scaled = scaler.transform(input_data)
        user_scaled_df = pd.DataFrame(input_scaled, columns=chart_names)

        prediction = model.predict(input_scaled)[0]
        proba_all = model.predict_proba(input_scaled)[0]
        prob = float(proba_all[int(prediction)])

        st.markdown(f"## {texts['prediction_result']}")
        if prediction == 1:
            result_color = "#28a745"
            result_icon = "✅"
            result_text = texts["healthy"]
        else:
            result_color = "#dc3545"
            result_icon = "⚠️"
            result_text = texts["unhealthy"]

        st.markdown(f"""
        <div style="background: {result_color}; color: white; padding: 2rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
            <h2 style="margin: 0; font-size: 2rem;">{result_icon} {result_text}</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">{texts['confidence']}: <strong>{prob:.4f}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"## {texts['feature_importance']}")
        final_model = model.steps[-1][1] if hasattr(model, "steps") else model
        if hasattr(final_model, "feature_importances_"):
            feature_importance = final_model.feature_importances_
            features = chart_names
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(features, feature_importance, color=["#ff6b6b", "#4ecdc4", "#45b7d1"])
            ax.set_xlabel("Importance", fontsize=12)
            ax.set_title("Feature Importance Analysis", fontsize=14, fontweight="bold")
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height() / 2,
                        f"{width:.3f}", ha="left", va="center", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown(f"## {texts['shap_plot']}")
        if not SHAP_AVAILABLE:
            st.info(texts["shap_unavailable"])
        else:
            try:
                explainer = shap.Explainer(model.predict_proba, background_data)
                shap_values = explainer(user_scaled_df)
                background_predictions = model.predict_proba(background_data)
                expected_value = background_predictions.mean(axis=0)

                if hasattr(shap_values, "values"):
                    if len(shap_values.values.shape) == 3:
                        shap_vals = shap_values.values[0, :, 1]
                        base_val = expected_value[1]
                    else:
                        shap_vals = shap_values.values[0, :]
                        base_val = expected_value[0]
                else:
                    shap_vals = shap_values[0, :]
                    base_val = expected_value[0]

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(texts["base_value"], f"{base_val:.4f}")
                with col_b:
                    st.metric(texts["final_prediction"], f"{base_val + shap_vals.sum():.4f}")

                with st.expander(texts["expand_shap"], expanded=True):
                    try:
                        plt.figure(figsize=(20, 8))
                        shap.force_plot(
                            base_val, shap_vals,
                            user_scaled_df.iloc[0],
                            feature_names=chart_names,
                            matplotlib=True, show=False,
                        )
                        plt.title("SHAP Force Plot - Current Prediction", fontsize=16, fontweight="bold", pad=30)
                        plt.tight_layout()
                        st.pyplot(plt)
                        plt.close()
                        st.success(texts["shap_success"])
                    except Exception as e:
                        st.warning(f"Matplotlib version failed: {e}")
                        try:
                            force_plot = shap.force_plot(
                                base_val, shap_vals,
                                user_scaled_df.iloc[0],
                                feature_names=chart_names,
                                matplotlib=False,
                            )
                            components.html(shap.getjs() + force_plot.html(), height=400)
                            st.success(texts["shap_html_success"])
                        except Exception as e2:
                            st.warning(f"HTML version also failed: {e2}")
                            fig, ax = plt.subplots(figsize=(15, 8))
                            features = chart_names
                            feature_values = user_scaled_df.iloc[0].values
                            colors = ["#ff6b6b" if x < 0 else "#4ecdc4" for x in shap_vals]
                            bars = ax.barh(features, shap_vals, color=colors, alpha=0.8, height=0.6)
                            for bar, shap_val, feature_val, feature_name in zip(bars, shap_vals, feature_values, features):
                                width = bar.get_width()
                                y_pos = bar.get_y() + bar.get_height() / 2
                                ax.text(width / 2, y_pos, f"{shap_val:.3f}",
                                        ha="center", va="center", color="white", fontweight="bold", fontsize=12)
                                if width > 0:
                                    ax.text(width + 0.05, y_pos, f"{feature_name}: {feature_val:.2f}",
                                            ha="left", va="center", fontsize=11, fontweight="bold")
                                else:
                                    ax.text(width - 0.05, y_pos, f"{feature_name}: {feature_val:.2f}",
                                            ha="right", va="center", fontsize=11, fontweight="bold")
                            ax.axvline(x=0, color="black", linestyle="-", alpha=0.5, linewidth=2)
                            ax.set_xlabel("SHAP Value")
                            ax.set_title("SHAP Force Plot - Feature Contributions")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            st.success(texts["shap_custom_success"])
            except Exception as e:
                st.error(f"{texts['shap_failed']}: {e}")
                st.info(texts["shap_unavailable"])
    except Exception as e:
        st.error(f"{texts['prediction_failed']}: {e}")

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 2rem 0; color: #666;">
    <p style="margin: 0;">{texts['footer']}</p>
</div>
""", unsafe_allow_html=True)
