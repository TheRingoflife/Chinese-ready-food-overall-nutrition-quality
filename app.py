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
        "title": "🍱 HSR-derived class prototype",
        "subtitle": "On-pack approximation of Health Star Rating outputs",
        "description": "Frozen XGBoost pipeline from the revised analysis. It approximates HSR < 3.5 vs ≥ 3.5 from three on-pack variables. It is not an independent healthfulness score or consumer advice.",
        "target_audience": "🎯 Intended use",
        "audience_desc": "Research demonstration where full HSR inputs are not routinely labelled.",
        "problem_statement": "📊 What it predicts",
        "problem_desc": "HSR-derived binary class: Unhealthy (HSR < 3.5) vs Healthy (HSR ≥ 3.5).",
        "solution": "💡 Locked inputs",
        "solution_desc": "Sodium (mg/100 g), Protein (g/100 g), Energy (kJ/100 g). No ultra-processed dummy in this binary model.",
        "mission": "⚠️ Limits",
        "mission_desc": "Not an official Health Star Rating, not Nutri-Score, and not dietary advice. Enter pack values unscaled.",
        "input_variables": "🔢 Input variables",
        "sodium_label": "Sodium (mg/100g)",
        "protein_label": "Protein (g/100g)",
        "energy_label": "Energy (kJ/100g)",
        "predict_button": "🧮 Predict HSR-derived class",
        "prediction_result": "🔍 Prediction result",
        "healthy": "✅ Healthy (HSR ≥ 3.5)",
        "unhealthy": "⚠️ Unhealthy (HSR < 3.5)",
        "confidence": "Predicted-class probability",
        "feature_importance": "📊 Feature importance",
        "shap_plot": "📊 SHAP force plot",
        "base_value": "Base value",
        "final_prediction": "Final prediction",
        "expand_shap": "Click to view SHAP force plot",
        "shap_success": "✅ SHAP force plot created (Matplotlib)!",
        "shap_html_success": "✅ SHAP force plot created (HTML backup)!",
        "shap_custom_success": "✅ SHAP bar plot with feature names!",
        "shap_table": "📊 SHAP values table",
        "shap_table_info": "💡 SHAP values displayed as a table",
        "positive_impact": "Toward Healthy (HSR ≥ 3.5)",
        "negative_impact": "Toward Unhealthy (HSR < 3.5)",
        "warning_input": "⚠️ Please enter pack values before predicting.",
        "input_tip": "💡 Enter values as printed on the pack. Do not standardise. Order: Sodium, Protein, Energy.",
        "model_error": "❌ Cannot proceed without model, scaler2 and background_data files",
        "prediction_failed": "Prediction failed",
        "shap_failed": "SHAP analysis failed",
        "shap_unavailable": "💡 SHAP explanation is not available, but feature importance is shown above.",
        "footer": "Research-use prototype · Streamlit + XGBoost · not for consumer advice.",
        "feature_names": ["Sodium", "Protein", "Energy"],
        "chart_feature_names": ["Sodium", "Protein", "Energy"],
    },
    "zh": {
        "title": "🍱 HSR 导出类别近似",
        "subtitle": "用包装标签变量近似 Health Star Rating 二分类",
        "description": "修订分析冻结的 XGBoost。用三个包装变量近似 HSR < 3.5 与 ≥ 3.5。不是独立的健康性评分，也不是消费建议。",
        "target_audience": "🎯 用途",
        "audience_desc": "研究演示：在无法常规获得完整 HSR 输入时展示有限标签信息的近似能力。",
        "problem_statement": "📊 预测什么",
        "problem_desc": "HSR 导出二分类：Unhealthy（HSR < 3.5）与 Healthy（HSR ≥ 3.5）。",
        "solution": "💡 锁定输入",
        "solution_desc": "钠 (mg/100 g)、蛋白质 (g/100 g)、能量 (kJ/100 g)。本二分类模型不含超加工哑变量。",
        "mission": "⚠️ 边界",
        "mission_desc": "不是官方 Health Star Rating，不是 Nutri-Score，也不是膳食建议。输入包装原值，不要预先标准化。",
        "input_variables": "🔢 输入变量",
        "sodium_label": "钠 (mg/100g)",
        "protein_label": "蛋白质 (g/100g)",
        "energy_label": "能量 (kJ/100g)",
        "predict_button": "🧮 预测 HSR 导出类别",
        "prediction_result": "🔍 预测结果",
        "healthy": "✅ 健康类 (HSR ≥ 3.5)",
        "unhealthy": "⚠️ 不健康类 (HSR < 3.5)",
        "confidence": "预测类别概率",
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
        "positive_impact": "推向健康类 (HSR ≥ 3.5)",
        "negative_impact": "推向不健康类 (HSR < 3.5)",
        "warning_input": "⚠️ 请先输入包装标示值。",
        "input_tip": "💡 按包装原样输入，不要标准化。顺序：钠、蛋白质、能量。",
        "model_error": "❌ 没有模型、scaler2 和 background_data 无法继续",
        "prediction_failed": "预测失败",
        "shap_failed": "SHAP分析失败",
        "shap_unavailable": "💡 SHAP解释不可用，但上面显示了特征重要性。",
        "footer": "研究演示原型 · Streamlit + XGBoost · 不得作为消费建议。",
        "feature_names": ["钠", "蛋白质", "能量"],
        "chart_feature_names": ["Sodium", "Protein", "Energy"],
    },
}

st.set_page_config(
    page_title="HSR-derived class prototype",
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
        # Locked order: Sodium, Protein, Energy. Unscaled pack values.
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
