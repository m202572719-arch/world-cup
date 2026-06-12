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
    st.error("Gemini API 客户端初始化失败，请检查云端 Secrets 配置。")

# ==========================================
# 2. 2026美加墨世界杯：官方正赛 48 强高精度量化数据库（严格对照分组图）
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
    "巴西": {"Elo": 2080, "Att": 1.38, "Def": 0.85, "Pedigree": 1.30, "Alt_Fit": True, "Style": "五星桑巴技术细腻，前场天才爆发力顶级，但近期防后防有隐患。"},
    "摩洛哥": {"Elo": 1940, "Att": 1.20, "Def": 0.80, "Pedigree": 1.10, "Alt_Fit": False, "Style": "北非铁血防线，退防密不透风，边路就地传切反击速度极快。"},
    "海地": {"Elo": 1580, "Att": 0.92, "Def": 1.02, "Pedigree": 1.00, "Alt_Fit": False, "Style": "附加赛黑马，球员爆发力和拼抢凶狠度强，但防守缺乏层次。"},
    "苏格兰": {"Elo": 1780, "Att": 1.04, "Def": 0.88, "Pedigree": 1.00, "Alt_Fit": False, "Style": "典型英伦硬朗风格，中场就地缠斗绞杀强，意志力极为坚韧。"},

    # --- Group D ---
    "美国": {"Elo": 1850, "Att": 1.15, "Def": 0.88, "Pedigree": 1.05, "Alt_Fit": False, "Style": "东道主，留洋青年军，主场冲击力和高频压迫极具侵略性。"},
    "巴拉圭": {"Elo": 1740, "Att": 0.92, "Def": 0.88, "Pedigree": 1.00, "Alt_Fit": True, "Style": "南美著名的低位硬骨头，死守反击能力强，球风极其极其凶悍。"},
    "澳大利亚": {"Elo": 1785, "Att": 1.04, "Def": 0.90, "Pedigree": 1.00, "Alt_Fit": False, "Style": "澳洲袋鼠身体强壮，高空争顶、定位球及长传砸禁区是杀手锏。"},
    "土耳其": {"Elo": 1845, "Att": 1.18, "Def": 0.88, "Pedigree": 1.00, "Alt_Fit": False, "Style": "星月军团作风彪悍，前场青年天才爆破力强，擅长打对攻乱战。"},

    # --- Group E ---
    "德国": {"Elo": 1980, "Att": 1.28, "Def": 0.88, "Pedigree": 1.25, "Alt_Fit": False, "Style": "日耳曼战车重回稳健，强调中场控制与战术纪律，整体向前推进。"},
    "库拉索": {"Elo": 1550, "Att": 0.90, "Def": 1.05, "Pedigree": 1.00, "Alt_Fit": False, "Style": "大黑马，多名归化坐镇，具备突出的单兵身体素质。"},
    "科特迪瓦": {"Elo": 1795, "Att": 1.14, "Def": 0.91, "Pedigree": 1.05, "Alt_Fit": False, "Style": "非洲大象身体素质拉满，中后场防守拦截硬度高，冲击力极强. "},
    "厄瓜多尔": {"Elo": 1870, "Att": 1.08, "Def": 0.83, "Pedigree": 1.00, "Alt_Fit": True, "Style": "高原体能怪，中场疯狗逼抢，两翼边路插上飞快。"},

    # --- Group F ---
    "荷兰": {"Elo": 1950, "Att": 1.22, "Def": 0.79, "Pedigree": 1.15, "Alt_Fit": False, "Style": "顶级中卫群领衔防线，全攻全守底蕴，但缺乏绝对尖刀中锋。"},
    "日本": {"Elo": 1925, "Att": 1.26, "Def": 0.84, "Pedigree": 1.05, "Alt_Fit": False, "Style": "亚洲地面传控天花板，高频就地反抢和小组传导配合极其娴熟。"},
    "瑞典": {"Elo": 1855, "Att": 1.20, "Def": 0.88, "Pedigree": 1.10, "Alt_Fit": False, "Style": "北欧力量与技术的结合，锋线球星极具终结力，攻防转换极快。"},
    "突尼斯": {"Elo": 1760, "Att": 0.98, "Def": 0.88, "Pedigree": 1.00, "Alt_Fit": False, "Style": "北非纪律流，极度擅长摆大巴、中场长线死守绞杀，十分顽固。"},

    # --- Group G ---
    "比利时": {"Elo": 1905, "Att": 1.25, "Def": 0.89, "Pedigree": 1.05, "Alt_Fit": False, "Style": "新老交替的欧洲红魔，进攻组织依旧犀利，但中后防线怕速度冲击。"},
    "埃及": {"Elo": 1775, "Att": 1.12, "Def": 0.91, "Pedigree": 1.00, "Alt_Fit": False, "Style": "全队立足坚固防反，依赖前场核心巨星抓反击失误一针见血。"},
    "伊朗": {"Elo": 1840, "Att": 1.12, "Def": 0.86, "Pedigree": 1.05, "Alt_Fit": False, "Style": "波斯铁骑，亚洲身体对抗之王，前场高塔抢点终结力极高。"},
    "新西兰": {"Elo": 1620, "Att": 0.94, "Def": 0.96, "Pedigree": 1.00, "Alt_Fit": False, "Style": "传统英式长传轰炸，身材高大对抗好，但脚下移动偏慢。"},

    # --- Group H ---
    "西班牙": {"Elo": 2045, "Att": 1.42, "Def": 0.82, "Pedigree": 1.20, "Alt_Fit": False, "Style": "极致传控配合，窒息的高位压迫，两侧年轻边锋极具爆破力。"},
    "佛得角": {"Elo": 1650, "Att": 0.95, "Def": 0.94, "Pedigree": 1.00, "Alt_Fit": False, "Style": "技术细腻的非洲神秘之师，战术灵活，擅长打就地防守反击。"},
    "沙特阿拉伯": {"Elo": 1695, "Att": 0.96, "Def": 0.96, "Pedigree": 1.00, "Alt_Fit": False, "Style": "脚下技术出色，纪律性好，离开西亚客战时战力打八折。"},
    "乌拉圭": {"Elo": 1960, "Att": 1.30, "Def": 0.82, "Pedigree": 1.20, "Alt_Fit": True, "Style": "疯狗式中场绞杀与狂野反击结合，作风极其彪悍顽强。"},

    # --- Group I ---
    "法国": {"Elo": 2110, "Att": 1.52, "Def": 0.78, "Pedigree": 1.25, "Alt_Fit": False, "Style": "核武器级别的防守反击，两翼速度恐怖，中场硬度极高。"},
    "塞内加尔": {"Elo": 1865, "Att": 1.18, "Def": 0.85, "Pedigree": 1.05, "Alt_Fit": False, "Style": "非洲特兰加雄狮，三线球星坐镇，爆发力与对抗力量无敌。"},
    "伊拉克": {"Elo": 1670, "Att": 0.98, "Def": 0.97, "Pedigree": 1.00, "Alt_Fit": False, "Style": "中东强悍球风，作风剽悍，身体对抗好，依靠高空球抢点。"},
    "挪威": {"Elo": 1835, "Att": 1.24, "Def": 0.89, "Pedigree": 1.00, "Alt_Fit": False, "Style": "前场拥有超级大杀器中锋，反击爆破和终结效率极高。"},

    # --- Group J ---
    "阿根廷": {"Elo": 2140, "Att": 1.45, "Def": 0.72, "Pedigree": 1.30, "Alt_Fit": False, "Style": "卫冕冠军，传控、逼抢与心理素质完美，核心谢幕战精神力拉满。"},
    "阿尔及利亚": {"Elo": 1810, "Att": 1.15, "Def": 0.90, "Pedigree": 1.00, "Alt_Fit": False, "Style": "北非传统技术流，脚下结合快，前场小范围地面渗透出色。"},
    "奥地利": {"Elo": 1835, "Att": 1.16, "Def": 0.87, "Pedigree": 1.00, "Alt_Fit": False, "Style": "极为极端的运动量逼抢派，高位压迫疯狂，攻防转换奇快。"},
    "约旦": {"Elo": 1690, "Att": 0.98, "Def": 0.95, "Pedigree": 1.00, "Alt_Fit": False, "Style": "亚洲新贵硬骨头，全队退防速度快，擅长打硬仗死守。"},

    # --- Group K ---
    "葡萄牙": {"Elo": 2010, "Att": 1.32, "Def": 0.84, "Pedigree": 1.10, "Alt_Fit": False, "Style": "阵容极度豪华且均衡，巨星单兵终结能力顶级，擅长快速推进。"},
    "乌兹别克斯坦": {"Elo": 1745, "Att": 1.02, "Def": 0.90, "Pedigree": 1.00, "Alt_Fit": False, "Style": "中亚白狼，身体极为强壮，战术极其硬朗，中后场防守稳固。"},
    "哥伦比亚": {"Elo": 1930, "Att": 1.24, "Def": 0.84, "Pedigree": 1.05, "Alt_Fit": True, "Style": "狂野南美流，脚下技术与强悍对抗完美结合，近期状态火爆。"},
    "民主刚果": {"Elo": 1675, "Att": 0.98, "Def": 0.96, "Pedigree": 1.00, "Alt_Fit": False, "Style": "典型的非洲力量对抗流，前场靠乱战，防守硬度大。"},

    # --- Group L ---
    "英格兰": {"Elo": 2050, "Att": 1.35, "Def": 0.80, "Pedigree": 1.15, "Alt_Fit": False, "Style": "身价全球第一，边路突破与阵地轰炸能力顶级，作风严谨。"},
    "克罗地亚": {"Elo": 1910, "Att": 1.12, "Def": 0.83, "Pedigree": 1.15, "Alt_Fit": False, "Style": "魔笛领衔格子军团，控节奏顶级，大赛心理与韧性极度恐怖。"},
    "加纳": {"Elo": 1720, "Att": 1.06, "Def": 0.98, "Pedigree": 1.00, "Alt_Fit": False, "Style": "阵中多位英超妖星，身体天赋爆表，但防守组织经常散架。"},
    "巴拿马": {"Elo": 1710, "Att": 0.98, "Def": 0.92, "Pedigree": 1.00, "Alt_Fit": False, "Style": "中北美坚韧反击流，阵型层次紧凑，反击落点非常明确。"}
}

GLOBAL_AVG_GOALS = 1.35

# ==========================================
# 3. 动态全维度精算模型 (支持主客场双向东道主检测)
# ==========================================
def calculate_advanced_match(team_A, team_B, is_high_altitude, squad_integrity_A, squad_integrity_B):
    data_A, data_B = TEAM_DATABASE[team_A], TEAM_DATABASE[team_B]
    att_A, def_A = data_A["Att"], data_A["Def"]
    att_B, def_B = data_B["Att"], data_B["Def"]
    
    # 伤病完整度折损系数
    att_A *= (squad_integrity_A / 100.0)
    att_B *= (squad_integrity_B / 100.0)
    
    lambda_A = att_A * def_B * GLOBAL_AVG_GOALS
    lambda_B = att_B * def_A * GLOBAL_AVG_GOALS
    
    # 【全面修正】：差异化判断主场或客场的东道主权重
    if team_A == "美国":
        lambda_A *= 1.15  # 美国主场哨与美式场馆系数
    elif team_A in ["墨西哥", "加拿大"]:
        lambda_A *= 1.12  # 墨西哥、加拿大常规主场加成
        
    if team_B == "美国":
        lambda_B *= 1.15
    elif team_B in ["墨西哥", "加拿大"]:
        lambda_B *= 1.12
        
    # 高海拔地缘缺氧环境特殊折损 (仅针对墨西哥赛区激活)
    if is_high_altitude:
        if not data_A["Alt_Fit"]:
            lambda_A *= 0.92
        if not data_B["Alt_Fit"]:
            lambda_B *= 0.92
            
    # 计算双泊松离散概率矩阵
    max_goals = 6
    score_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            score_matrix[i, j] = poisson.pmf(i, lambda_A) * poisson.pmf(j, lambda_B)
            
    prob_A_win = float(np.sum(np.tril(score_matrix, -1)))
    prob_draw = float(np.sum(np.diag(score_matrix)))
    prob_B_win = float(np.sum(np.triu(score_matrix, 1)))
    
    # 世界杯冠军DNA（Pedigree）权重博弈干预
    pedigree_gap = data_A["Pedigree"] - data_B["Pedigree"]
    if pedigree_gap > 0:
        prob_A_win += (pedigree_gap * 0.1)
        prob_B_win -= (pedigree_gap * 0.1)
    elif pedigree_gap < 0:
        prob_B_win += (abs(pedigree_gap) * 0.1)
        prob_A_win -= (abs(pedigree_gap) * 0.1)
        
    # 归一化重整
    total = prob_A_win + prob_draw + prob_B_win
    prob_A_win, prob_draw, prob_B_win = prob_A_win/total, prob_draw/total, prob_B_win/total
    
    # 提炼比分波胆前三名
    flat_indices = np.argsort(score_matrix.ravel())[::-1][:3]
    top_scores = []
    for idx in flat_indices:
        i, j = divmod(idx, max_goals)
        top_scores.append((f"{i}:{j}", score_matrix[i, j] / np.sum(score_matrix)))
        
    return prob_A_win, prob_draw, prob_B_win, lambda_A, lambda_B, top_scores

# ==========================================
# 4. Streamlit 交互层
# ==========================================
st.set_page_config(page_title="2026世界杯精算推演器", page_icon="🏆", layout="wide")

st.title("🏆 2026美加墨世界杯：48强正赛足彩精密辅助系统")
st.markdown("⚠️ **数据安全重校版本：** 完美接入高级 3.5 Flash 开发接口。数学矩阵已双向绑定美、加、墨三大东道主独立主场盘口权重。")
st.divider()

st.sidebar.header(f"📊 官方正赛 48 强精准量化看板")
sidebar_df = pd.DataFrame.from_dict(TEAM_DATABASE, orient='index')[['Elo', 'Att', 'Def', 'Pedigree', 'Alt_Fit']]
st.sidebar.dataframe(sidebar_df, height=600)

st.subheader("🛠️ 彩票临场变数调节（结合突发受伤、停赛、球场海拔）")
col_env1, col_env2, col_env3 = st.columns(3)
with col_env1:
    is_altitude = st.checkbox("🏔️ 设定本场在墨西哥阿兹特克等高海拔缺氧球场（非高原队获进球期望扣减）")
with col_env2:
    integrity_A = st.slider("🎯 主队临场核心完整度 (%)", 50, 100, 100)
with col_env3:
    integrity_B = st.slider("🛡️ 客队临场核心完整度 (%)", 50, 100, 100)

st.divider()

col_a, col_b = st.columns(2)
with col_a:
    team_A = st.selectbox("🎯 选择主队 (Team A)", list(TEAM_DATABASE.keys()), index=4)  # 默认加拿大
with col_b:
    team_B = st.selectbox("🛡️ 选择客队 (Team B)", list(TEAM_DATABASE.keys()), index=7)  # 默认波黑

st.info(f"💡 **主队盘口实力分析 ({team_A})：** {TEAM_DATABASE[team_A]['Style']}")
st.info(f"💡 **客队盘口实力分析 ({team_B})：** {TEAM_DATABASE[team_B]['Style']}")

if st.button("🔥 运行泊松矩阵进行足彩盘口精密推演", use_container_width=True):
    if team_A == team_B:
        st.warning("⚠️ 相同球队无法交锋，请重新挑选对手。")
    else:
        p_A, p_draw, p_B, exp_A, exp_B, top_scores = calculate_advanced_match(
            team_A, team_B, is_altitude, integrity_A, integrity_B
        )
        
        st.subheader("📊 独家足彩胜平负、比分精算期望")
        res_1, res_2, res_3 = st.columns(3)
        res_1.metric(f"【胜】{team_A} 胜率", f"{p_A:.2%}", f"期望进球: {exp_A:.2f}")
        res_2.metric("【平】平局概率", f"{p_draw:.2%}")
        res_3.metric(f"【负】{team_B} 胜率", f"{p_B:.2%}", f"期望进球: {exp_B:.2f}")
        
        st.progress(int(p_A * 100), text=f"{team_A} 独赢胜出概率空间分布")
        
        st.markdown("##### 🎯 精确波胆（比分）几率前三高预测：")
        score_text = " ｜ ".join([f"预测 **{score}** (精确几率 {prob:.1%})" for score, prob in top_scores])
        st.write(score_text)
        st.divider()
        
        # ==========================================
        # 5. 调用大模型生成战术策略报告（对接 3.5 Flash 正规生产接口字符串）
        # ==========================================
        st.subheader("🧠 Gemini 3.5 旗舰级足彩战术博弈深度报告")
        with st.spinner("🤖 正在调度 3.5 Flash 专家思考内核结合主场优势进行硬核预测..."):
            
            pedigree_A = TEAM_DATABASE[team_A]["Pedigree"]
            pedigree_B = TEAM_DATABASE[team_B]["Pedigree"]
            
            prompt = f"""
            你是一位享誉全球的硬核足球足彩精算大师，行文风格锐利、专业、极具博弈论视角的金钱说服力。
            请针对这场2026世界杯焦点大战进行足彩下注层面的战术推演：{team_A} VS {team_B}。
            
            后端双泊松精算模型给出的确定性上下文如下：
            1. 精算胜率：{team_A}胜率 {p_A:.1%}，平局率 {p_draw:.1%}，{team_B}胜率 {p_B:.1%}。
            2. 期望进球（XG）：{team_A}为 {exp_A:.2f}，{team_B}为 {exp_B:.2f}。
            3. 赛场环境变量：高海拔缺氧球场={is_altitude}。
            4. 动态战力完整度：{team_A}={integrity_A}% ｜ {team_B}={integrity_B}%。
            5. 精确比分概率前三名为：{top_scores[0][0]}、{top_scores[1][0]}、{top_scores[2][0]}。
            6. 战术本底：
               - {team_A}（底蕴权重 {pedigree_A}）：{TEAM_DATABASE[team_A]['Style']}
               - {team_B}（底蕴权重 {pedigree_B}）：{TEAM_DATABASE[team_B]['Style']}
            
            请严格结合场地、美国（1.15）/加拿大（1.12）/墨西哥（1.12）差异化的主场特权特性能量、巨星残损率，撰写一份包含以下模块的足彩内参：
            - 【主场优势与控场】：深度拆解主客场地利（包含美国的高分贝或加拿大的快草皮）对当前盘口克制关系的影响。
            - 【数据合理性拆解】：结合XG和基本盘，向彩民解释为什么模型会得出这样的胜率与比分期望。
            - 【足彩X因素防范】：直接指出哪些突发变数会颠覆这个冷冰冰的数学模型。
            字数控制在 400 字以内，直击痛点，一针见血。
            """
            try:
                # 对接 3.5 世代官方生产标准模型字符串：'gemini-3.5-flash'
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=prompt,
                )
                st.write(response.text)
            except Exception as e:
                st.error(f"3.5 Flash 智能策略生成失败，错误信息: {e}")
