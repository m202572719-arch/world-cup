import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
import google.genai as genai

# ==========================================
# 1. 初始化 Gemini 客户端
# ==========================================
try:
    # 自动从环境或 Streamlit Secrets 读取密钥，严防 GitHub 拦截
    client = genai.Client()
except Exception as e:
    st.error("Gemini API 客户端初始化失败，请检查云端 Secrets 配置。")

# ==========================================
# 2. 2026美加墨世界杯：48强严格量化数据库
# ==========================================
# 维度：Elo(积分), Att(进攻), Def(防守), Pedigree(底蕴权重1.0~1.3), Altitude_Fit(高原耐受), Style(风格)
TEAM_DATABASE = {
    # --- 欧洲区 ---
    "阿根廷": {"Elo": 2140, "Att": 1.45, "Def": 0.72, "Pedigree": 1.30, "Alt_Fit": False,
               "Style": "卫冕冠军，传控与前场逼抢结合，梅西谢幕战精神属性拉满。"},
    "法国": {"Elo": 2110, "Att": 1.52, "Def": 0.78, "Pedigree": 1.25, "Alt_Fit": False,
             "Style": "顶级防反，锋线具备恐怖绝对速度，中场拦截硬度极高。"},
    "英格兰": {"Elo": 2050, "Att": 1.35, "Def": 0.80, "Pedigree": 1.15, "Alt_Fit": False,
               "Style": "身价极高，阵地战与边路冲击力强，大赛作风偏向稳健。"},
    "西班牙": {"Elo": 2045, "Att": 1.42, "Def": 0.82, "Pedigree": 1.20, "Alt_Fit": False,
               "Style": "极致传控，高位压迫令人窒息，两侧年轻边锋冲击力强。"},
    "葡萄牙": {"Elo": 2010, "Att": 1.32, "Def": 0.84, "Pedigree": 1.10, "Alt_Fit": False,
               "Style": "三线均衡，反击推进极快，阵中球星具备极强单兵终结能力。"},
    "德国": {"Elo": 1980, "Att": 1.28, "Def": 0.88, "Pedigree": 1.25, "Alt_Fit": False,
             "Style": "战术纪律性极强，注重整体推进与中场控制，处于新老交替期。"},
    "荷兰": {"Elo": 1950, "Att": 1.22, "Def": 0.79, "Pedigree": 1.15, "Alt_Fit": False,
             "Style": "全攻全守传统，顶级中卫领衔后防，但锋线缺乏绝对尖刀。"},
    "意大利": {"Elo": 1920, "Att": 1.15, "Def": 0.81, "Pedigree": 1.20, "Alt_Fit": False,
               "Style": "传统链式防守底蕴，中场绞杀能力强，球风坚韧但锋线较弱。"},
    "克罗地亚": {"Elo": 1910, "Att": 1.12, "Def": 0.83, "Pedigree": 1.15, "Alt_Fit": False,
                 "Style": "格子军团韧性恐怖，中场控节奏顶级，极擅长加时赛消耗战。"},
    "比利时": {"Elo": 1905, "Att": 1.25, "Def": 0.89, "Pedigree": 1.05, "Alt_Fit": False,
               "Style": "红魔新老交替，进攻依靠核心串联，后防面对速度型前锋吃力。"},
    "瑞士": {"Elo": 1880, "Att": 1.10, "Def": 0.84, "Pedigree": 1.05, "Alt_Fit": False,
             "Style": "战术执行力极高的硬骨头，整体纪律严明，擅长掀翻豪强。"},
    "丹麦": {"Elo": 1860, "Att": 1.12, "Def": 0.86, "Pedigree": 1.05, "Alt_Fit": False,
             "Style": "典型北欧力量型足球，定位球战术极具威胁，组织井井有条。"},
    "乌克兰": {"Elo": 1820, "Att": 1.14, "Def": 0.89, "Pedigree": 1.00, "Alt_Fit": False,
               "Style": "前场具备极强个人反击爆发力，斗志高昂，擅长乱战反击。"},
    "奥地利": {"Elo": 1835, "Att": 1.16, "Def": 0.87, "Pedigree": 1.00, "Alt_Fit": False,
               "Style": "擅长疯狂的运动量逼抢与高位压迫，攻防转换速度极快。"},
    "波兰": {"Elo": 1780, "Att": 1.08, "Def": 0.92, "Pedigree": 1.00, "Alt_Fit": False,
             "Style": "围绕核心中锋展开攻势，高空球能力强，但中场控制力偏弱。"},
    "捷克": {"Elo": 1795, "Att": 1.10, "Def": 0.90, "Pedigree": 1.00, "Alt_Fit": False,
             "Style": "身体对抗能力极强，球风大开大合，依靠边路传中制造威胁。"},

    # --- 南美区 ---
    "巴西": {"Elo": 2080, "Att": 1.38, "Def": 0.85, "Pedigree": 1.30, "Alt_Fit": True,
             "Style": "技术细腻，前场天才云集，边路突破凌厉，但近期防守有隐患。"},
    "乌拉圭": {"Elo": 1960, "Att": 1.30, "Def": 0.82, "Pedigree": 1.20, "Alt_Fit": True,
               "Style": "球风狂野高压，疯狗式中场绞杀与快速纵深反击结合，作风凶狠。"},
    "哥伦比亚": {"Elo": 1930, "Att": 1.24, "Def": 0.84, "Pedigree": 1.05, "Alt_Fit": True,
                 "Style": "身体素质与技术完美结合，阵地战创造力强，近期状态极其火热。"},
    "厄瓜多尔": {"Elo": 1870, "Att": 1.08, "Def": 0.83, "Pedigree": 1.00, "Alt_Fit": True,
                 "Style": "典型高原雄鹰，奔跑能力恐怖，防守硬朗，反击边路速度极快。"},
    "秘鲁": {"Elo": 1750, "Att": 1.00, "Def": 0.94, "Pedigree": 1.00, "Alt_Fit": True,
             "Style": "传统南美脚法，注重整体传控配合，但面对高压逼抢易失误。"},
    "巴拉圭": {"Elo": 1740, "Att": 0.92, "Def": 0.88, "Pedigree": 1.00, "Alt_Fit": True,
               "Style": "防守型球队，作风顽强，擅长死守反击以及利用定位球偷袭。"},

    # --- 中北美及加勒比区 (含三大东道主) ---
    "美国": {"Elo": 1850, "Att": 1.15, "Def": 0.88, "Pedigree": 1.05, "Alt_Fit": False,
             "Style": "东道主，阵容年轻且多在欧洲效力，速度快冲击力强，主场加成大。"},
    "墨西哥": {"Elo": 1820, "Att": 1.12, "Def": 0.90, "Pedigree": 1.05, "Alt_Fit": True,
               "Style": "东道主，美洲技术流，高海拔魔鬼主场结合极快脚下传切。"},
    "加拿大": {"Elo": 1790, "Att": 1.14, "Def": 0.93, "Pedigree": 1.00, "Alt_Fit": False,
               "Style": "东道主，拥有世界级边路超跑，反击推进速度极快，后防稚嫩。"},
    "哥斯达黎加": {"Elo": 1690, "Att": 0.95, "Def": 0.95, "Pedigree": 1.00, "Alt_Fit": False,
                   "Style": "擅长低位防守陷阱，大赛经验丰富，极其坚韧的老牌反击队。"},
    "巴拿马": {"Elo": 1710, "Att": 0.98, "Def": 0.92, "Pedigree": 1.00, "Alt_Fit": False,
               "Style": "中北美异军突起的黑马，球员身体强壮，快速反击套路成熟。"},
    "牙买加": {"Elo": 1705, "Att": 1.05, "Def": 0.91, "Pedigree": 1.00, "Alt_Fit": False,
               "Style": "雷鬼男孩身体素质劲爆，拥有多名英超速度流前锋，边路撕扯强。"},

    # --- 非洲区 ---
    "摩洛哥": {"Elo": 1940, "Att": 1.20, "Def": 0.80, "Pedigree": 1.10, "Alt_Fit": False,
               "Style": "纯粹足球，防守组织密不透风，边路反击与小组配合顶级，韧性极强。"},
    "塞内加尔": {"Elo": 1865, "Att": 1.18, "Def": 0.85, "Pedigree": 1.05, "Alt_Fit": False,
                 "Style": "特兰加雄狮，三线均有顶级球星，身体对抗与爆发力极其恐怖。"},
    "突尼斯": {"Elo": 1760, "Att": 0.98, "Def": 0.88, "Pedigree": 1.00, "Alt_Fit": False,
               "Style": "北非纪律流，球风偏向防守和中场缠斗，节奏较慢但防守十分顽固。"},
    "阿尔及利亚": {"Elo": 1810, "Att": 1.15, "Def": 0.90, "Pedigree": 1.00, "Alt_Fit": False,
                   "Style": "技术细腻的前场攻击群，擅长地面渗透，但中后防高空球较弱。"},
    "埃及": {"Elo": 1775, "Att": 1.12, "Def": 0.91, "Pedigree": 1.00, "Alt_Fit": False,
             "Style": "围绕超级边锋展开的防守反击战术，反击落点明确，打法硬朗。"},
    "尼日利亚": {"Elo": 1800, "Att": 1.22, "Def": 0.95, "Pedigree": 1.00, "Alt_Fit": False,
                 "Style": "前场拥有顶级神锋攻击群，爆发力无敌，但门将与后防不稳定。"},
    "喀麦隆": {"Elo": 1730, "Att": 1.05, "Def": 0.96, "Pedigree": 1.00, "Alt_Fit": False,
               "Style": "非洲传统劲旅，作风勇猛彪悍，擅长长传背身策应，防守大开大合。"},
    "加纳": {"Elo": 1720, "Att": 1.06, "Def": 0.98, "Pedigree": 1.00, "Alt_Fit": False,
             "Style": "阵中多位年轻英超妖星，个人身体天赋爆表，但战术组织稍弱。"},
    "南非": {"Elo": 1680, "Att": 0.96, "Def": 0.94, "Pedigree": 1.00, "Alt_Fit": False,
             "Style": "以国内班底为主，默契度极高，擅长就地的小范围传导与控球。"},

    # --- 亚洲区 ---
    "日本": {"Elo": 1925, "Att": 1.26, "Def": 0.84, "Pedigree": 1.05, "Alt_Fit": False,
             "Style": "亚洲技术流天花板，全留洋明星阵容，地面传导与高位逼抢顶级。"},
    "伊朗": {"Elo": 1840, "Att": 1.12, "Def": 0.86, "Pedigree": 1.05, "Alt_Fit": False,
             "Style": "波斯铁骑，亚洲身体对抗之王，前场双子星终结力极强，打法铁血。"},
    "韩国": {"Elo": 1830, "Att": 1.15, "Def": 0.89, "Pedigree": 1.05, "Alt_Fit": False,
             "Style": "太极虎奔跑与体能极其疯狂，前场拥有顶级球星闪光点，后防易失误。"},
    "澳大利亚": {"Elo": 1785, "Att": 1.04, "Def": 0.90, "Pedigree": 1.00, "Alt_Fit": False,
                 "Style": "袋鼠军团走英式路线，身体强壮、高空球轰炸与定位球是杀手锏。"},
    "沙特阿拉伯": {"Elo": 1695, "Att": 0.96, "Def": 0.96, "Pedigree": 1.00, "Alt_Fit": False,
                   "Style": "脚下技术出色，擅长整体造越位战术，离开西亚后战力稍打折扣。"},
    "卡塔尔": {"Elo": 1715, "Att": 1.02, "Def": 0.95, "Pedigree": 1.00, "Alt_Fit": False,
               "Style": "亚洲杯冠军，归化体系成熟，前场连线默契度极高，擅长打防反。"},
    "伊拉克": {"Elo": 1670, "Att": 0.98, "Def": 0.97, "Pedigree": 1.00, "Alt_Fit": False,
               "Style": "球风凶悍剽悍，身体对抗强，门前抢点与头球攻门能力不俗。"},
    "阿曼": {"Elo": 1640, "Att": 0.92, "Def": 0.95, "Pedigree": 1.00, "Alt_Fit": False,
             "Style": "典型的中东紧凑流派，战术极其重视防守阵型的层次感，反击坚决。"},

    # --- 大洋洲与附加赛 ---
    "新西兰": {"Elo": 1620, "Att": 0.94, "Def": 0.96, "Pedigree": 1.00, "Alt_Fit": False,
               "Style": "大洋洲霸主，英式传统长传轰炸打法，后防高大但转身移动较慢。"},
    "海地": {"Elo": 1580, "Att": 0.92, "Def": 1.02, "Pedigree": 1.00, "Alt_Fit": False,
             "Style": "通过附加赛突围，黑人身体素质极佳，踢法极具侵略性和未知性。"},
    "古拉索": {"Elo": 1550, "Att": 0.90, "Def": 1.05, "Pedigree": 1.00, "Alt_Fit": False,
               "Style": "本届超级新面孔，多名荷兰血统归化，球风兼具身体与欧化打法。"}
}

GLOBAL_AVG_GOALS = 1.35


# ==========================================
# 3. 动态全维度精算数学模型
# ==========================================
def calculate_advanced_match(team_A, team_B, is_high_altitude, squad_integrity_A, squad_integrity_B):
    # 提取基础系数
    data_A, data_B = TEAM_DATABASE[team_A], TEAM_DATABASE[team_B]
    att_A, def_A = data_A["Att"], data_A["Def"]
    att_B, def_B = data_B["Att"], data_B["Def"]

    # 动态修正 1: 考虑伤病/核心缺阵对进攻系数的直接折损
    att_A *= (squad_integrity_A / 100.0)
    att_B *= (squad_integrity_B / 100.0)

    # 计算初始泊松期望进球数值
    lambda_A = att_A * def_B * GLOBAL_AVG_GOALS
    lambda_B = att_B * def_A * GLOBAL_AVG_GOALS

    # 动态修正 2: 东道主加成权重 (美、墨、加)
    hosts = ["美国", "墨西哥", "加拿大"]
    if team_A in hosts:
        lambda_A *= 1.12
    if team_B in hosts:
        lambda_B *= 1.12

    # 动态修正 3: 极端高海拔缺氧生存环境折损
    if is_high_altitude:
        if not data_A["Alt_Fit"]:
            lambda_A *= 0.92  # 非高原型球队进球期望暴跌 8%
        if not data_B["Alt_Fit"]:
            lambda_B *= 0.92

    # 构建双泊松矩阵
    max_goals = 6
    score_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            score_matrix[i, j] = poisson.pmf(i, lambda_A) * poisson.pmf(j, lambda_B)

    # 提炼基础胜平负空间
    prob_A_win = float(np.sum(np.tril(score_matrix, -1)))
    prob_draw = float(np.sum(np.diag(score_matrix)))
    prob_B_win = float(np.sum(np.triu(score_matrix, 1)))

    # 动态修正 4: 强队淘汰赛/大赛精神Pedigree博弈加权
    pedigree_gap = data_A["Pedigree"] - data_B["Pedigree"]
    if pedigree_gap > 0:
        prob_A_win += (pedigree_gap * 0.1)
        prob_B_win -= (pedigree_gap * 0.1)
    elif pedigree_gap < 0:
        prob_B_win += (abs(pedigree_gap) * 0.1)
        prob_A_win -= (abs(pedigree_gap) * 0.1)

    # 归一化确保概率总和严格等于 1.0
    total = prob_A_win + prob_draw + prob_B_win
    prob_A_win, prob_draw, prob_B_win = prob_A_win / total, prob_draw / total, prob_B_win / total

    # 提取精确比分前三名
    flat_indices = np.argsort(score_matrix.ravel())[::-1][:3]
    top_scores = []
    for idx in flat_indices:
        i, j = divmod(idx, max_goals)
        top_scores.append((f"{i}:{j}", score_matrix[i, j] / np.sum(score_matrix)))

    return prob_A_win, prob_draw, prob_B_win, lambda_A, lambda_B, top_scores


# ==========================================
# 4. Streamlit 渲染层
# ==========================================
st.set_page_config(page_title="2026世界杯全栈精密推演器", page_icon="🏆", layout="wide")

st.title("🏆 2026美加墨世界杯：48强全维度工业级智能推演网")
st.markdown("本系统已进入终极形态！深度融合**双泊松矩阵、东道主天时、高海拔耐缺氧因数、伤病残损率**及大模型战术推演。")
st.divider()

# 侧边栏：完整量化数据面板验证
st.sidebar.header(f"📊 48强严谨量化看板 (当前: {len(TEAM_DATABASE)}队)")
sidebar_df = pd.DataFrame.from_dict(TEAM_DATABASE, orient='index')[['Elo', 'Att', 'Def', 'Pedigree', 'Alt_Fit']]
st.sidebar.dataframe(sidebar_df, height=500)

# 动态物理变动参数交互区
st.subheader("🛠️ 赛前动态物理变量配置（提高模型实时精准度）")
col_env1, col_env2, col_env3 = st.columns(3)
with col_env1:
    is_altitude = st.checkbox("🏔️ 设定本场为高海拔赛区（如墨西哥城、瓜达拉哈拉等阿兹特克缺氧球场）")
with col_env2:
    integrity_A = st.slider("🎯 主队阵容伤病完整度 (%)", 50, 100, 100, help="如有巨星伤停，请向左拉低系数")
with col_env3:
    integrity_B = st.slider("🛡️ 客队阵容伤病完整度 (%)", 50, 100, 100, help="如有核心红牌停赛，请向左拉低系数")

st.divider()

# 对阵选择区
col_a, col_b = st.columns(2)
with col_a:
    team_A = st.selectbox("🎯 选择主队 (Team A)", list(TEAM_DATABASE.keys()), index=0)
    st.info(f"**底层战术标签：** {TEAM_DATABASE[team_A]['Style']}")
with col_b:
    team_B = st.selectbox("🛡️ 选择客队 (Team B)", list(TEAM_DATABASE.keys()), index=1)
    st.info(f"**底层战术标签：** {TEAM_DATABASE[team_B]['Style']}")

# 运行复合推演
if st.button("🔥 启动工业级高胜率复合兵盘推演", use_container_width=True):
    if team_A == team_B:
        st.warning("⚠️ 相同球队无法交锋，请重新挑选对手。")
    else:
        p_A, p_draw, p_B, exp_A, exp_B, top_scores = calculate_advanced_match(
            team_A, team_B, is_altitude, integrity_A, integrity_B
        )

        st.subheader("📊 多维度数学精算结果 (Integrated Poisson Matrix)")
        res_1, res_2, res_3 = st.columns(3)
        res_1.metric(f"{team_A} 单场胜率", f"{p_A:.2%}", f"修正后期望进球: {exp_A:.2f}")
        res_2.metric("平局概率（90分钟）", f"{p_draw:.2%}")
        res_3.metric(f"{team_B} 单场胜率", f"{p_B:.2%}", f"修正后期望进球: {exp_B:.2f}")

        st.progress(int(p_A * 100), text=f"{team_A} 胜出博弈空间分布")
        # ==========================================
        # 5. 调用大模型生成报告 (严格修正变量作用域)
        # ==========================================
        st.subheader("🧠 Gemini 3.5 旗舰级战术沙盘兵推报告")
        with st.spinner("🤖 正在召集人工智能足球专家团结合高海拔、伤情进行硬核复盘..."):

            # 严格从全局字典读取两队底蕴权重
            pedigree_A = TEAM_DATABASE[team_A]["Pedigree"]
            pedigree_B = TEAM_DATABASE[team_B]["Pedigree"]

            prompt = f"""
                    你是一位享誉全球的硬核足球战术精算大师，行文风格锐利、专业、充满数据和物理因数说服力。
                    请针对这场2026世界杯焦点战进行多维度战术复盘推演：{team_A} VS {team_B}。

                    后端高胜率精算模型给出的物理与数学上下文如下：
                    1. 最终修正胜率：{team_A}胜率 {p_A:.1%}，平局率 {p_draw:.1%}，{team_B}胜率 {p_B:.1%}。
                    2. 期望进球（XG）：{team_A}为 {exp_A:.2f}，{team_B}为 {exp_B:.2f}。
                    3. 赛场环境变量：高海拔缺氧球场={is_altitude}。
                    4. 动态伤情系数：{team_A}战力完整度 {integrity_A}% ｜ {team_B}战力完整度 {integrity_B}%。
                    5. 精确比分前三名为：{top_scores[0][0]}、{top_scores[1][0]}、{top_scores[2][0]}。
                    6. 战术本底：
                       - {team_A}（底蕴权重 {pedigree_A}）：{TEAM_DATABASE[team_A]['Style']}
                       - {team_B}（底蕴权重 {pedigree_B}）：{TEAM_DATABASE[team_B]['Style']}

                    请严格结合高海拔、完整度折损与数理胜率，撰写一份包含以下模块的战术推演报告：
                    - 【控场战术博弈】：分析两队风格碰撞在当前环境（如高原缺氧、巨星伤停）下会如何演变。
                    - 【数据合理性拆解】：解释为什么模型考虑了动态因数后会得出这样的胜率。
                    - 【胜负手与X因素】：指出哪一个细节会颠覆这个数学模型。
                    字数控制在 400 字以内，一针见血。
                    """

            try:
                # 严格调度 3.5 旗舰大模型进行深度逻辑复盘
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=prompt,
                )
                st.write(response.text)
            except Exception as e:
                st.error(f"大模型推演失败，错误信息: {e}")
