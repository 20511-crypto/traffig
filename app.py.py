import os
import pickle
import numpy as np
import streamlit as st
from Orange.data import Domain, Table

# ---------------------------------------------------------
# 1. 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="교통사고 예상 사망자 수 예측 및 스쿨존 안전 대시보드",
    page_icon="🚦",
    layout="wide",
)

# ---------------------------------------------------------
# 2. Orange3 모델 로드 (캐싱 적용으로 빠른 실시간 추론)
# ---------------------------------------------------------
MODEL_PATH = "traffic_casualty_model.pkcls"


@st.cache_resource
def load_orange_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


try:
    model = load_orange_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ 모델 로드 중 오류가 발생했습니다: {e}")
    st.info("GitHub 저장소 루트 경로에 `traffic_casualty_model.pkcls` 파일이 있는지 확인해주세요.")
    st.stop()

# 모델의 원본 입력 도메인에서 시도(DiscreteVariable) 목록 자동 추출
orig_domain = model.original_domain
feature_attrs = list(orig_domain.attributes)
feature_domain = Domain(feature_attrs)

sido_options = ["경기", "부산", "서울"]
for attr in feature_attrs:
    if attr.name == "시도" and hasattr(attr, "values"):
        sido_options = list(attr.values)
        break

# ---------------------------------------------------------
# 3. 실시간 예측 함수
# ---------------------------------------------------------
def predict_fatalities(
    sido: str,
    accidents: float,
    serious_injuries: float,
    minor_injuries: float,
    reported_injuries: float,
) -> float:
    row_values = []
    for attr in feature_attrs:
        if attr.name == "사고건수":
            row_values.append(float(accidents))
        elif attr.name == "중상자수":
            row_values.append(float(serious_injuries))
        elif attr.name == "경상자수":
            row_values.append(float(minor_injuries))
        elif attr.name == "부상신고자수":
            row_values.append(float(reported_injuries))
        elif attr.name == "시도":
            row_values.append(attr.to_val(sido))
        else:
            row_values.append(np.nan)

    input_table = Table.from_list(feature_domain, [row_values])
    pred = model(input_table)
    # 회귀 모델 음수 예측 방지를 위해 0 이상으로 보정
    return max(0.0, float(pred[0]))


# ---------------------------------------------------------
# 4. UI 헤더 & 입력 컨트롤 (변경 시 즉시 재실행)
# ---------------------------------------------------------
st.title("🚦 교통사고 위험도 실시간 예측 & 어린이 보호구역 안전 가이드")
st.caption("Orange3 Neural Network 모델(`traffic_casualty_model.pkcls`) 기반 실시간 시뮬레이터")

col_input, col_result = st.columns([1.05, 1.25], gap="large")

with col_input:
    st.subheader("1️⃣ 지역 및 사고 지표 입력")

    selected_sido = st.selectbox(
        "📍 시도 선택",
        options=sido_options,
        index=sido_options.index("서울") if "서울" in sido_options else 0,
        help="모델이 학습한 시도 범주(경기, 부산, 서울) 중 하나를 선택합니다.",
    )

    accidents = st.slider(
        "🚗 사고건수 (건)",
        min_value=50,
        max_value=4500,
        value=1000,
        step=10,
    )

    minor_injuries = st.slider(
        "🩹 경상자수 (명)",
        min_value=50,
        max_value=5500,
        value=1100,
        step=10,
    )

    # 모델 학습에 포함된 나머지 2개 피처(중상자수, 부상신고자수) 처리
    with st.expander("⚙️ 상세 부상 지표 설정 (모델 입력 호환용)", expanded=False):
        auto_ratio = st.checkbox(
            "사고건수 비례 자동 추정 사용 (권장)",
            value=True,
            help="체크 시 학습 데이터의 평균 비율(중상자 약 30%, 부상신고자 약 13%)을 자동 반영합니다.",
        )
        if auto_ratio:
            serious_injuries = round(accidents * 0.30, 1)
            reported_injuries = round(accidents * 0.13, 1)
            st.write(f"- 자동 반영 **중상자수**: `{serious_injuries:,.1f}명`")
            st.write(f"- 자동 반영 **부상신고자수**: `{reported_injuries:,.1f}명`")
        else:
            serious_injuries = st.slider("🚑 중상자수 (명)", 10, 1500, int(accidents * 0.30), 5)
            reported_injuries = st.slider("📋 부상신고자수 (명)", 0, 1000, int(accidents * 0.13), 5)

# 슬라이더/셀렉트박스 조작 즉시 실시간 예측 실행
predicted_deaths = predict_fatalities(
    sido=selected_sido,
    accidents=accidents,
    serious_injuries=serious_injuries,
    minor_injuries=minor_injuries,
    reported_injuries=reported_injuries,
)

# ---------------------------------------------------------
# 5. 위험도 판정 및 대형 신호등 + 어린이 보호구역 안전 수칙 카드
# ---------------------------------------------------------
if predicted_deaths < 5.0:
    status_key = "SAFE"
    status_badge = "🟢 안전 (SAFE)"
    status_color = "#16a34a"
    bg_tint = "#f0fdf4"
    border_color = "#86efac"
    status_desc = "예상 사망자 수가 낮은 수준입니다. 현재의 스쿨존 감속 및 보행 안전 수칙을 꾸준히 유지해주세요."
    school_rules = [
        ("시속 30km 이하 서행 준수", "어린이 보호구역 내 제한속도를 준수하고 전방 시야를 확보하세요."),
        ("횡단보도 일시정지 생활화", "보행자가 없더라도 스쿨존 내 무신호 횡단보도 앞에서는 반드시 일시정지합니다."),
        (" 불법 주·정차 절대 금지", "주정차 차량은 어린이의 돌발 횡단 시 운전자의 시야를 가리는 주원인입니다."),
    ]
elif predicted_deaths < 15.0:
    status_key = "CAUTION"
    status_badge = "🟡 주의 (CAUTION)"
    status_color = "#d97706"
    bg_tint = "#fffbeb"
    border_color = "#fcd34d"
    status_desc = "중대 인명피해 발생 가능성이 감지됩니다. 등·하교 시간대 집중 주의와 방어운전이 필요합니다."
    school_rules = [
        ("등·하교 시간대(08~10시, 13~16시) 집중 감속", "시속 20~30km 이하로 서행하며 골목길·통학로 진입 시 경적을 자제하세요."),
        ("우회전 시 횡단보도 앞 완전 정지", "우회전 직전과 직후 횡단보도 보행자 유무를 2회 이상 확인 후 통과하세요."),
        ("어린이 통학버스 특별 보호", "정차 중인 통학버스 추월을 금지하고, 점멸등 작동 시 일시정지 후 서행하세요."),
    ]
else:
    status_key = "DANGER"
    status_badge = "🔴 위험 (DANGER)"
    status_color = "#dc2626"
    bg_tint = "#fef2f2"
    border_color = "#fca5a5"
    status_desc = "예상 사망자 수가 높은 고위험 상태입니다! 스쿨존 내 각별한 감속 운전과 교통안전 관리가 시급합니다."
    school_rules = [
        ("스쿨존 진입 전 사전 감속 & 방어운전", "어린이는 키가 작아 사각지대에 가려지기 쉬우므로 시속 20km 수준으로 서행하세요."),
        ("스마트폰·내비게이션 조작 절대 금지", "전방 주시 태만은 치명적 사고로 직결됩니다. 주행 중 조작을 완전히 멈추세요."),
        ("어린이 보행자 안전 수칙 지도", "횡단보도 앞 '멈추기 → 좌우 살피기 → 손 들고 건너기' 및 무단횡단 금지를 철저히 실천하세요."),
    ]

red_active = "opacity: 1; box-shadow: 0 0 32px #ef4444;" if status_key == "DANGER" else "opacity: 0.22;"
yellow_active = "opacity: 1; box-shadow: 0 0 32px #f59e0b;" if status_key == "CAUTION" else "opacity: 0.22;"
green_active = "opacity: 1; box-shadow: 0 0 32px #22c55e;" if status_key == "SAFE" else "opacity: 0.22;"

with col_result:
    st.subheader("2️⃣ 실시간 예측 결과 및 위험도 신호등")

    # 예측 수치 메트릭 + 대형 신호등 HTML 렌더링
    traffic_light_html = f"""
    <div style="
        background-color: {bg_tint};
        border: 2px solid {border_color};
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        flex-wrap: wrap;
    ">
        <div style="flex: 1; min-width: 220px;">
            <div style="font-size: 0.95rem; color: #475569; font-weight: 600;">
                [{selected_sido}] 실시간 예상 사망자 수
            </div>
            <div style="font-size: 3.1rem; font-weight: 800; color: {status_color}; line-height: 1.15; margin: 6px 0;">
                {predicted_deaths:.2f} <span style="font-size: 1.3rem; font-weight: 600;">명</span>
            </div>
            <div style="
                display: inline-block;
                background-color: {status_color};
                color: white;
                font-weight: 700;
                font-size: 1.1rem;
                padding: 6px 16px;
                border-radius: 999px;
                margin-bottom: 10px;
            ">
                {status_badge}
            </div>
            <p style="margin: 6px 0 0 0; color: #334155; font-size: 0.95rem; line-height: 1.5;">
                {status_desc}
            </p>
        </div>

        <!-- 대형 3구 신호등 -->
        <div style="
            background-color: #1e293b;
            padding: 16px 20px;
            border-radius: 28px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 14px;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.6), 0 8px 18px rgba(15,23,42,0.2);
        ">
            <div style="width: 68px; height: 68px; border-radius: 50%; background: #ef4444; {red_active} display:flex; align-items:center; justify-content:center; color:white; font-weight:800; font-size:0.85rem;">위험</div>
            <div style="width: 68px; height: 68px; border-radius: 50%; background: #f59e0b; {yellow_active} display:flex; align-items:center; justify-content:center; color:white; font-weight:800; font-size:0.85rem;">주의</div>
            <div style="width: 68px; height: 68px; border-radius: 50%; background: #22c55e; {green_active} display:flex; align-items:center; justify-content:center; color:white; font-weight:800; font-size:0.85rem;">안전</div>
        </div>
    </div>
    """
    st.markdown(traffic_light_html, unsafe_allow_html=True)

    # 어린이 보호구역 안전 수칙 카드
    rules_items_html = "".join(
        [
            f"""
            <div style="
                background: #ffffff;
                border-left: 5px solid {status_color};
                border-radius: 10px;
                padding: 12px 16px;
                margin-bottom: 10px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);
            ">
                <div style="font-weight: 700; color: #0f172a; font-size: 1.0rem; margin-bottom: 4px;">
                    ✅ {title}
                </div>
                <div style="color: #475569; font-size: 0.92rem; line-height: 1.45;">
                    {desc}
                </div>
            </div>
            """
            for title, desc in school_rules
        ]
    )

    safety_card_html = f"""
    <div style="
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
    ">
        <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 12px;">
            🚸 어린이 보호구역(스쿨존) 필수 안전 수칙 카드
        </div>
        {rules_items_html}
    </div>
    """
    st.markdown(safety_card_html, unsafe_allow_html=True)