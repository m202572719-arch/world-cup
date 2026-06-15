import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
import google.genai as genai

# ==========================================
# 1. 初始化 Gemini 客户端
# ==========================================
try:
    client = genai.Client()
except Exception as e:
    st.error("Gemini API 客户端初始化失败，请检查云端 Secrets配置。")

# ==========================================
# 2. 2026美加墨世界杯：官方正赛 48 强量化数据库
# ==========================================
TEAM_DATABASE = {
    # --- Group A ---
    "墨西哥": {"Elo": 1820, "Att": 1.12, "Def": 0.90, "Pedigree": 1.05, "Alt_Fit": True, "Style": "东道主，中美洲技术流，阿兹特克高原魔鬼主场，脚下传切快。"},
    "南非": {"Elo": 1680, "Att": 0.96, "Def": 0.94, "Pedigree": 1.00, "Alt_Fit": False, "Style": "反击推进快，依靠整体就地小范围传导，但缺乏锋线强力终结者。"},
    "韩国": {"Elo": 1830, "Att": 1.15, "Def": 0.89, "Pedigree": 1.05, "Alt_Fit": False, "Style": "太极虎高位奔跑极其疯狂，前场巨星闪光爆发力强。"},
    "捷克": {"Elo": 1795, "Att": 1.10, "Def": 0.90, "Pedigree": 1.00, "Alt_Fit": False, "Style": "身体对抗极强，擅长高空球轰炸与两翼边路传中。"},

    # --- Group B ---
    "加拿大": {"Elo": 1790, "Att": 1.14, "Def": 0.93, "Pedigree": 1.00, "Alt_Fit": False, "Style": "东道主，两翼速度极快，纵深推进能力强，后防稍显年轻。"},
    "瑞士": {"Elo": 1880, "Att": 1.10, "Def": 0.84, "Pedigree": 1.05, "Alt_Fit": False, "Style": "战术执行力极高的硬骨头，整体防守非常严密，纪律性极强。"},
    "卡塔尔": {"Elo": 1715, "Att": 1.02, "Def": 0.95, "Pedigree": 1.00, "Alt_Fit": False, "Style": "亚洲杯常客，传控默契度高，极度依赖前场的反击效率。"},
    "波黑": {"Elo": 1715, "Att": 1.02, "Def": 0.92, "Pedigree": 1.00, "Alt_Fit": False, "Style": "欧陆力量流派，身材高大，极其擅长定位球乱战与禁区砸头球。"},

    # --- Group C ---
    "巴西": {"Elo": 2080, "Att": 1.38, "Def": 0.85, "Pedigree": 1.30, "Alt_Fit": True, "Style": "五星桑巴技术细腻，前场天才爆发力顶级，但近期后防有隐患。"},
    "摩洛哥": {"Elo": 1940, "Att": 1.20, "Def": 0.80, "Pedigree": 1.10, "Alt_Fit": False, "Style": "北非铁血防线，退防密不透风，边路就地传切反击速度极快。"},
    "海地": {"Elo": 1580, "Att": 0.92, "Def": 1.02, "Pedigree": 1.00, "Alt_Fit": False, "Style": "附加赛黑马，球员爆发力和拼抢凶狠度强，但防守缺乏层次。"},
    "苏格兰": {"Elo": 1780, "Att": 1.04, "Def": 0.88, "Pedigree": 1.00, "Alt_Fit": False, "Style": "典型英伦硬朗风格，中场就地缠斗绞杀强，意志力极为坚韧。"},

    # --- Group D ---
    "美国": {"Elo": 1850, "Att": 1.15, "Def": 0.88, "Pedigree": 1.05, "Alt_Fit": False, "Style": "东道主，留洋青年军，主场冲击力和高频压迫极具侵略性。"},
    "巴拉圭": {"Elo": 1740, "Att": 0.92, "Def": 0.88, "Pedigree": 1.00, "Alt_Fit": True, "Style": "南美著名的低位硬骨头，死守反击能力强，球风极其凶悍。"},
    "澳大利亚": {"Elo": 1785, "Att": 1.04, "Def": 0.90, "Pedigree": 1.00, "Alt_Fit": False, "Style": "澳洲袋鼠身体强壮，高空争顶、定位球及长传砸禁区是杀手锏。"},
    "土耳其": {"Elo": 1845, "Att": 1.18, "Def": 0.88, "Pedigree": 1.00, "Alt_Fit": False, "Style": "星月军团作风彪悍，前场青年天才爆破力强，擅长打对攻乱战。"},

    # --- Group E ---
    "德国": {"Elo": 1980, "Att": 1.28, "Def": 0.88, "Pedigree": 1.25, "Alt_Fit": False, "Style": "日耳曼战车重回稳健，强调中场控制与战术纪律，整体向前推进。"},
    "库拉索": {"Elo": 1550, "Att": 0.90, "Def": 1.05, "Pedigree": 1.00, "Alt_Fit": False, "Style": "大黑马，多名归化坐镇，具备突出的单兵身体素质。"},
    "科特迪瓦": {"Elo": 1795, "Att": 1.14, "Def": 0.91, "Pedigree": 1.05, "Alt_Fit": False, "Style": "非洲大象身体素质拉满，中后场防守拦截硬度高，冲击力极强。"},
    "厄瓜多尔": {"Elo": 1870, "Att": 1.08, "Def": 0.83, "Pedigree": 1.00, "Alt_Fit": True, "Style": "高原体能怪，中场疯狗逼抢，两翼边路插上飞快。"},

    # --- Group L ---
    "英格兰": {"Elo": 2050, "Att": 1.35, "Def": 0.80, "Pedigree": 1.15, "Alt_Fit": False, "Style": "身价全球第一，边路突破与阵地轰炸能力顶级，作风严谨。"},
    "克罗地亚": {"Elo": 1910, "Att": 1.12, "Def": 0.83, "Pedigree": 1.15, "Alt_Fit": False, "Style": "魔笛领衔格子军团，控节奏顶级，大赛心理与韧性极度恐怖。"}
}

GLOBAL_AVG_GOALS = 1.35

# ==========================================
# 3. 动态打破平滑的精密数学引擎
# ==========================================
def calculate_advanced_match(team_A, team_B, venue_type, integrity_A, integrity_B, aggression_factor):
    data_A, data_B = TEAM_DATABASE[team_A], TEAM_DATABASE[team_B]
    att_A, def_A = data_A["Att"], data_A["Def"]
    att_B, def_B = data_B["Att"], data_B["Def"]
    
    att_A *= (integrity_A / 100.0)
    att_B *= (integrity_B / 100.0)
    
    # 核心改动：激进因子（Aggression Factor）直接放大期望进球基数，强行拉大方差
    lambda_A = att_A * def_B * GLOBAL_AVG_GOALS * aggression_factor
    lambda_B = att_B * def_A * GLOBAL_AVG_GOALS * aggression_factor
    
    if venue_type == "美国主场（NFL大型场馆 & 高分贝判罚优势）" and team_A == "美国":
        lambda_A *= 1.15
    elif venue_type == "加拿大主场（高纬度低温 & 人工合成快草皮）" and team_A == "加拿大":
        lambda_A *= 1.12
    elif venue_type == "墨西哥主场（2200米阿兹特克高原缺氧生态）":
        if team_A == "墨西哥": lambda_A *= 1.12
        if not data_A["Alt_Fit"]: lambda_A *= 0.92
        if not data_B["Alt_Fit"]: lambda_B *= 0.92
            
    # 计算 6x6 概率空间矩阵
    max_goals = 6
    score_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            score_matrix[i, j] = poisson.pmf(i, lambda_A) * poisson.pmf(j, lambda_B)
    
    # 【狄克森-科尔修正流派微调】：如果模型极度激进，人为对 0:0, 1:1 进行低比分抑制系数压制
    if aggression_factor > 1.1:
        score_matrix[0, 0] *= 0.75
        score_matrix[1, 1] *= 0.80
        score_matrix[0, 1] *= 0.85
        score_matrix[1, 0] *= 0.85
        # 重新归一化
        score_matrix /= np.sum(score_matrix)

    prob_A_win = float(np.sum(np.tril(score_matrix, -1)))
    prob_draw = float(np.sum(np.diag(score_matrix)))
    prob_B_win = float(np.sum(np.triu(score_matrix, 1)))
    
    # 归一化重整胜平负
    total = prob_A_win + prob_draw + prob_B_win
    prob_A_win, prob_draw, prob_B_win = prob_A_win/total, prob_draw/total, prob_B_win/total
    
    # 提取真正的大比分波胆前三
    flat_indices = np.argsort(score_matrix.ravel())[::-1][:3]
    top_scores = []
    for idx in flat_indices:
        i, j = divmod(idx, max_goals)
        top_scores.append((f"{i}:{j}", score_matrix[i, j]))
        
    return prob_A_win, prob_draw, prob_B_win, lambda_A, lambda_B, top_scores

# ==========================================
# 4. Streamlit 前端渲染
# ==========================================
st.set_page_config(page_title="2026世界杯精算推演器", page_icon="🏆", layout="wide")
st.title("🏆 2026世界杯：反过度平滑（Anti-Smoothing）足彩精算辅助系统")

st.subheader("🛠️ 核心建模变数控制台")
col_env1, col_env2 = st.columns([2, 2])
with col_env1:
    venue = st.radio("🏟️ 设定本场赛地的地缘环境因子", ["中立场地 / 其他常规赛区", "美国主场（NFL大型场馆）", "加拿大主场（人工合成快草皮）", "墨西哥主场（2200米阿兹特克高原）"], index=2)
with col_env2:
    # 💥 这是解开过度平滑的终极武器！
    agg_factor = st.slider("🔥 战术博弈激进烈度（强行打破平滑均值，拉大比分方差）", 0.8, 1.8, 1.3, step=0.1, help="拉到1.3以上，系统将压制低平比分，强制精算高进球乱战、大球波胆！")

st.divider()

col_a, col_b = st.columns(2)
with col_a: team_A = st.selectbox("🎯 选择主队 (Team A)", list(TEAM_DATABASE.keys()), index=4)
with col_b: team_B = st.selectbox("🛡️ 选择客队 (Team B)", list(TEAM_DATABASE.keys()), index=7)

if st.button("🔥 运行去平滑泊松矩阵进行精密推演", use_container_width=True):
    p_A, p_draw, p_B, exp_A, exp_B, top_scores = calculate_advanced_match(team_A, team_B, venue, 100, 100, agg_factor)
    
    st.subheader("📊 独家足彩胜平负、比分精算期望")
    res_1, res_2, res_3 = st.columns(3)
    res_1.metric(f"【胜】{team_A}", f"{p_A:.2%}", f"去平滑期望进球: {exp_A:.2f}")
    res_2.metric("【平】平局概率", f"{p_draw:.2%}")
    res_3.metric(f"【负】{team_B}", f"{p_B:.2%}", f"去平滑期望进球: {exp_B:.2f}")
    
    st.markdown("##### 🎯 精确波胆（比分）几率预测（已启动 Dixon-Coles 抑制）：")
    score_text = " ｜ ".join([f"预测 **{score}** (精确几率 {prob:.1%})" for score, prob in top_scores])
    st.write(score_text)
    st.divider()
    
    st.subheader("🧠 Gemini 工业级足彩战术博弈深度内参")
    with st.spinner("🤖 正在调用最新稳定接口..."):
        prompt = f"针对 {team_A} VS {team_B}。期望进球：{team_A}={exp_A:.2f}, {team_B}={exp_B:.2f}。前三比分：{top_scores[0][0]}、{top_scores[1][0]}。激进烈度系数为 {agg_factor}。请用极度专业的足彩精算师口吻，给出斩钉截铁的【大小球盘口】与【独赢/让球走势】下注指导，400字以内。"
        try:
            response = genai.Client().models.generate_content(model='gemini-2.5-flash', contents=prompt)
            st.write(response.text)
        except Exception as e:
            st.error(f"大模型策略生成失败: {e}")
