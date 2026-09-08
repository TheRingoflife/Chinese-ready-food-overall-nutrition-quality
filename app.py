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
        "title": "🍱 Nutritional Quality Classifier",
        "subtitle": "Nutrition quality assessment based on HSR-generated labels",
        "description": "This advanced machine learning application uses XGBoost, based on key nutritional features, to perform nutrition quality assessment of ready foods based on HSR-generated labels. The main point is that a small set of routinely labelled nutrients can closely approximate overall HSR nutrition quality assessment results.",
        "target_audience": "🎯 Target Audience",
        "audience_desc": "Designed for countries with limited nutritional information where generalized positive labelling is difficult to implement.",
        "problem_statement": "📊 Problem Statement",
        "problem_desc": "Many countries lack comprehensive nutritional labeling systems, making it difficult to implement generalized positive labeling for food products.",
        "solution": "💡 Our Solution",
        "solution_desc": "The ML model analyzes 3 key nutritional features (sodium, protein, energy) to approximate the HSR-generated Healthy / Unhealthy class, with detailed explanations.",
        "mission": "🚀 Mission",
        "mission_desc": "Providing a practical approach for countries with incomplete nutritional information that cannot directly calculate a complete HSR: approximating HSR-generated nutrition quality assessment with a small set of on-pack nutrients. This is only a preliminary validation; current results reflect nutrition quality based on a partial set of nutrients.",
        "input_variables": "🔢 Input Variables",
        "protein_label": "Protein (g/100g)",
        "sodium_label": "Sodium (mg/100g)",
        "energy_label": "Energy (kJ/100g)",
        "predict_button": "🧮 Predict nutrition quality class",
        "prediction_result": "🔍 Prediction Result",
        "healthy": "✅ Healthy",
        "unhealthy": "⚠️ Unhealthy",
        "confidence": "Confidence",
        "feature_importance": "📊 Feature Importance",
        "shap_plot": "📊 SHAP Force Plot",
        "base_value": "Base value",
        "final_prediction": "Final prediction",
        "expand_shap": "Click to view SHAP force plot",
        "shap_success": "✅ SHAP force plot created (Matplotlib version)!",
        "shap_html_success": "✅ SHAP force plot created (HTML version - Backup)!",
        "shap_custom_success": "✅ SHAP force plot created (Custom version with feature names)!",
        "shap_table": "📊 SHAP Values Table",
        "shap_table_info": "💡 SHAP values displayed as table",
        "positive_impact": "Positive impact (toward Healthy class)",
        "negative_impact": "Negative impact (toward Unhealthy class)",
        "warning_input": "⚠️ Please enter values for at least one feature before predicting.",
        "input_tip": "💡 Tip: Please fill in according to the information on the product pack.",
        "model_error": "❌ Cannot proceed without model and scaler files",
        "prediction_failed": "Prediction failed",
        "shap_failed": "SHAP analysis failed",
        "shap_unavailable": "💡 SHAP explanation is not available, but feature importance is shown above.",
        "footer": "Developed using Streamlit and XGBoost · For research use only.",
        "feature_names": ["Sodium", "Protein", "Energy"],
        "chart_feature_names": ["Sodium", "Protein", "Energy"],
    },
    "zh": {
        "title": "🍱 营养质量分类器",
        "subtitle": "基于HSR生成的营养质量评估",
        "description": "这个先进的机器学习应用程序使用XGBoost根据关键营养特征，对预制食品进行基于HSR生成的营养质量评估。主要体现的是：少数常规标示营养素仍可较高程度近似HSR整体营养质量评估结果。",
        "target_audience": "🎯 目标用户",
        "audience_desc": "专为营养信息有限、难以实施概括性正面标签的国家设计。",
        "problem_statement": "📊 问题陈述",
        "problem_desc": "许多国家缺乏全面的营养标签系统，难以实施食品的概括性正面标签。",
        "solution": "💡 我们的解决方案",
        "solution_desc": "ML模型分析3个关键营养特征（钠、蛋白质、能量），近似HSR生成的健康/不健康类别，并给出详细解释。",
        "mission": "🚀 使命",
        "mission_desc": "为营养信息不全、无法直接计算完整HSR的国家提供一个使用思路：用少量包装营养素近似HSR生成的营养质量评估。当前只是初步验证，当前结果只是部分营养素上的营养质量。",
        "input_variables": "🔢 输入变量",
        "protein_label": "蛋白质 (g/100g)",
        "sodium_label": "钠 (mg/100g)",
        "energy_label": "能量 (kJ/100g)",
        "predict_button": "🧮 预测营养质量类别",
        "prediction_result": "🔍 预测结果",
        "healthy": "✅ 健康",
        "unhealthy": "⚠️ 不健康",
        "confidence": "置信度",
        "feature_importance": "📊 特征重要性",
        "shap_plot": "📊 SHAP力图",
        "base_value": "基准值",
        "final_prediction": "最终预测",
        "expand_shap": "点击查看SHAP力图",
        "shap_success": "✅ SHAP力图创建成功 (Matplotlib版本)!",
        "shap_html_success": "✅ SHAP力图创建成功 (HTML版本 - 备用)!",
        "shap_custom_success": "✅ SHAP力图创建成功 (自定义版本，包含特征名称)!",
        "shap_table": "📊 SHAP值表格",
        "shap_table_info": "💡 SHAP值以表格形式显示",
        "positive_impact": "积极影响（推向健康类）",
        "negative_impact": "消极影响（推向不健康类）",
        "warning_input": "⚠️ 请在预测前至少输入一个特征的值。",
        "input_tip": "💡 提示：请按照产品包装上信息填写。",
        "model_error": "❌ 没有模型和标准化器文件无法继续",
        "prediction_failed": "预测失败",
        "shap_failed": "SHAP分析失败",
        "shap_unavailable": "💡 SHAP解释不可用，但上面显示了特征重要性。",
        "footer": "使用Streamlit和XGBoost开发 · 仅供研究使用。",
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
