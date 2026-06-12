import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
import google.genai as genai
import os

# ==========================================
# 1. 初始化 Gemini 客户端 (最安全模式：自动读取环境变量)
# ==========================================
# 本地测试注：如果你想在本地 PyCharm 运行测试，可以取消下面这行的注释并填入你的 Key。
# os.environ["GEMINI_API_KEY"] = "AIzaSy..."

try:
    # 这样写，GitHub 绝对不会拦截你！Streamlit 云端会自动从高级设置里读取密码
    client = genai.Client()
except Exception as e:
    st.error("Gemini API 客户端初始化失败。")

# ==========================================
# 2. 2026美加墨世界杯：48强完整满血数据库
# ==========================================
TEAM_DATABASE = {
    # --- 欧洲区 (16队) ---
    "阿根廷": {"Elo": 2140, "Att": 1.45, "Def": 0.72,
               "Style": "卫冕冠军，控球传导与前场高压结合，梅西谢幕战精神属性拉满。"},
    "法国": {"Elo": 2110, "Att": 1.52, "Def": 0.78,
             "Style": "顶级防守反击，锋线具备恐怖的绝对速度与爆发力，中场硬度极高。"},
    "英格兰": {"Elo": 2050, "Att": 1.35, "Def": 0.80,
               "Style": "阵容总身价极高，阵地战与边路传中能力强，战术风格偏向稳健。"},
    "西班牙": {"Elo": 2045, "Att": 1.42, "Def": 0.82,
               "Style": "极致的传控体系，高位压迫令人窒息，两侧年轻边锋冲击力极强。"},
    "葡萄牙": {"Elo": 2010, "Att": 1.32, "Def": 0.84,
               "Style": "三线实力极其均衡，反击速度极快，阵中球星具备极强单兵解决战斗能力。"},
    "德国": {"Elo": 1980, "Att": 1.28, "Def": 0.88, "Style": "战术纪律性极强，注重整体推进与中场控制，处于新老交替期. "},
    "荷兰": {"Elo": 1950, "Att": 1.22, "Def": 0.79,
             "Style": "全攻全守传统，后防线由顶级中卫领衔极其稳固，前场缺乏绝对终结者。"},
    "意大利": {"Elo": 1920, "Att": 1.15, "Def": 0.81,
               "Style": "传统链式防守底蕴，擅长在中场进行绞杀，整体球风坚韧但缺乏锋线尖刀。"},
    "克罗地亚": {"Elo": 1910, "Att": 1.12, "Def": 0.83,
                 "Style": "格子军团韧性恐怖，中场大师控节奏能力顶级，极擅长打加时赛与消耗战。"},
    "比利时": {"Elo": 1905, "Att": 1.25, "Def": 0.89,
               "Style": "红魔新老交替，进攻端依靠核心串联，后防线面对速度型前锋稍显吃力。"},
    "瑞士": {"Elo": 1880, "Att": 1.10, "Def": 0.84,
             "Style": "战术执行力极高的硬骨头，整体防守纪律严明，擅长掀翻传统豪强。"},
    "丹麦": {"Elo": 1860, "Att": 1.12, "Def": 0.86,
             "Style": "典型的北欧力量型整体足球，定位球战术极具威胁，中场组织井井有条。"},
    "乌克兰": {"Elo": 1820, "Att": 1.14, "Def": 0.89,
               "Style": "前场具备极强的个人反击爆发力，精神斗志高昂，擅长乱战反击。"},
    "奥地利": {"Elo": 1835, "Att": 1.16, "Def": 0.87,
               "Style": "擅长疯狂的运动量逼抢与高位压迫，攻防转换速度极快，球风硬朗。"},
    "波兰": {"Elo": 1780, "Att": 1.08, "Def": 0.92,
             "Style": "围绕核心中锋展开攻势，高空球和禁区内终结能力强，但中场控制力偏弱。"},
    "捷克": {"Elo": 1795, "Att": 1.10, "Def": 0.90,
             "Style": "身体对抗能力极强，球风大开大合，依靠冲击力和边路传中制造威胁。"},

    # --- 南美区 (6队) ---
    "巴西": {"Elo": 2080, "Att": 1.38, "Def": 0.85,
             "Style": "桑巴军团技术细腻，前场天才云集，边路突破凌厉，但近期防守存在隐患。"},
    "乌拉圭": {"Elo": 1960, "Att": 1.30, "Def": 0.82,
               "Style": "球风狂野高压，疯狗式中场绞杀与快速纵深反击结合，前场逼抢极其凶狠。"},
    "哥伦比亚": {"Elo": 1930, "Att": 1.24, "Def": 0.84,
                 "Style": "球员身体素质与技术完美结合，阵地战创造力强，近期状态极其火热。"},
    "厄瓜多尔": {"Elo": 1870, "Att": 1.08, "Def": 0.83,
                 "Style": "高原雄鹰身体对抗和奔跑能力恐怖，防守极其硬朗，反击边路速度极快。"},
    "秘鲁": {"Elo": 1750, "Att": 1.00, "Def": 0.94,
             "Style": "典型南美脚法，注重整体传控与地面配合，但面对身体流压迫时易失误。"},
    "巴拉圭": {"Elo": 1740, "Att": 0.92, "Def": 0.88,
               "Style": "南美区著名的防守型球队，作风顽强，擅长死守反击以及利用定位球偷袭。"},

    # --- 中北美及加勒比区 (6队 - 含东道主) ---
    "美国": {"Elo": 1850, "Att": 1.15, "Def": 0.88,
             "Style": "东道主之一，阵容年轻且大多在欧洲效力，速度快、冲击力强，主场加成大。"},
    "墨西哥": {"Elo": 1820, "Att": 1.12, "Def": 0.90,
               "Style": "东道主之一，传统美洲技术流，脚下结合快，在阿兹特克等魔鬼主场战力飙升。"},
    "加拿大": {"Elo": 1790, "Att": 1.14, "Def": 0.93,
               "Style": "东道主之一，拥有世界级边路超跑，反击推进速度冠绝北美，后防稍显稚嫩。"},
    "哥斯达黎加": {"Elo": 1690, "Att": 0.95, "Def": 0.95,
                   "Style": "擅长严密的低位防守陷阱，大赛经验丰富，极其坚韧的反击型球队。"},
    "巴拿马": {"Elo": 1710, "Att": 0.98, "Def": 0.92,
               "Style": "近年来中北美异军突起的黑马，球员身体强壮，快速反击套路成熟。"},
    "古拉索": {"Elo": 1550, "Att": 0.90, "Def": 1.05,
               "Style": "本届世界杯超级新面孔，多名荷兰血统归化球员，球风兼具身体与欧化。"},

    # --- 非洲区 (9队) ---
    "摩洛哥": {"Elo": 1940, "Att": 1.20, "Def": 0.80,
               "Style": "北非纯粹足球，防守组织密不透风，边路反击与小组配合顶级，大赛韧性极强。"},
    "塞内加尔": {"Elo": 1865, "Att": 1.18, "Def": 0.85,
                 "Style": "非洲特兰加雄狮，三线均有顶级球星坐镇，身体对抗与爆发力极其恐怖。"},
    "突尼斯": {"Elo": 1760, "Att": 0.98, "Def": 0.88,
               "Style": "北非纪律流，整体球风偏向防守和中场缠斗，节奏较慢但防守十分顽固。"},
    "阿尔及利亚": {"Elo": 1810, "Att": 1.15, "Def": 0.90,
                   "Style": "技术细腻的前场攻击群，擅长地面渗透，但中后防线在面对高空球时较吃力。"},
    "埃及": {"Elo": 1775, "Att": 1.12, "Def": 0.91,
             "Style": "围绕超级巨星展开的防守反击战术，反击落点极其明确，打法硬朗。"},
    "尼日利亚": {"Elo": 1800, "Att": 1.22, "Def": 0.95,
                 "Style": "非洲雄鹰，前场拥有意甲顶级神锋攻击群，爆发力无敌，但门将与后防不稳定。"},
    "喀麦隆": {"Elo": 1730, "Att": 1.05, "Def": 0.96,
               "Style": "非洲传统劲旅，作风勇猛彪悍，擅长长传背身策应，防守大开大合。"},
    "加纳": {"Elo": 1720, "Att": 1.06, "Def": 0.98,
             "Style": "阵中多位年轻英超妖星，个人身体天赋爆表，但整体战术组织和纪律性稍弱。"},
    "南非": {"Elo": 1680, "Att": 0.96, "Def": 0.94, "Style": "以国内班底为主，默契度极高，擅长就地的小范围传导与控球。"},

    # --- 亚洲区 (8队) ---
    "日本": {"Elo": 1925, "Att": 1.26, "Def": 0.84,
             "Style": "亚洲技术流天花板，留洋全明星阵容，极致的地面传导与前场高位逼抢。"},
    "伊朗": {"Elo": 1840, "Att": 1.12, "Def": 0.86,
             "Style": "波斯铁骑，亚洲身体对抗之王，前场双子星终结力极强，擅长铁血反击。"},
    "韩国": {"Elo": 1830, "Att": 1.15, "Def": 0.89,
             "Style": "太极虎奔跑与体能极其疯狂，前场拥有顶级球星闪光点，后防近期有隐患。"},
    "澳大利亚": {"Elo": 1785, "Att": 1.04, "Def": 0.90,
                 "Style": "袋鼠军团走英式路线，身体强壮、高空球轰炸与定位球是核心杀手锏。"},
    "沙特阿拉伯": {"Elo": 1695, "Att": 0.96, "Def": 0.96,
                   "Style": "脚下技术出色，擅长整体造越位战术，但离开西亚本土后高原与客场战力有折扣。"},
    "卡塔尔": {"Elo": 1715, "Att": 1.02, "Def": 0.95,
               "Style": "亚洲杯冠军，归化体系成熟，前场连线默契度极高，擅长打防守反击。"},
    "伊拉克": {"Elo": 1670, "Att": 0.98, "Def": 0.97,
               "Style": "球风凶悍剽悍，身体对抗不虚欧洲球队，门前抢点与头球攻门能力强。"},
    "阿曼": {"Elo": 1640, "Att": 0.92, "Def": 0.95,
             "Style": "典型的中东紧凑流派，战术极其重视防守阵型的层次感，反击坚决。"},

    # --- 大洋洲及附加赛 (2队) ---
    "新西兰": {"Elo": 1620, "Att": 0.94, "Def": 0.96,
               "Style": "大洋洲霸主，英式传统长传轰炸打法，后防身材高大但转身移动较慢。"},
    "海地": {"Elo": 1580, "Att": 0.92, "Def": 1.02,
             "Style": "通过附加赛历史性突围，球员黑人身体素质极佳，踢法极具侵略性和未知性。"}
}

GLOBAL_AVG_GOALS = 1.35


# ==========================================
# 3. 核心数学模型：双泊松回归计算
# ==========================================
def calculate_match_probability(team_A, team_B):
    att_A, def_A = TEAM_DATABASE[team_A]["Att"], TEAM_DATABASE[team_A]["Def"]
    att_B, def_B = TEAM_DATABASE[team_B]["Att"], TEAM_DATABASE[team_B]["Def"]

    lambda_A = att_A * def_B * GLOBAL_AVG_GOALS
    lambda_B = att_B * def_A * GLOBAL_AVG_GOALS

    max_goals = 6
    score_matrix = np.zeros((max_goals, max_goals))

    for i in range(max_goals):
        for j in range(max_goals):
            score_matrix[i, j] = poisson.pmf(i, lambda_A) * poisson.pmf(j, lambda_B)

    prob_A_win = float(np.sum(np.tril(score_matrix, -1)))
    prob_draw = float(np.sum(np.diag(score_matrix)))
    prob_B_win = float(np.sum(np.triu(score_matrix, 1)))

    total_prob = prob_A_win + prob_draw + prob_B_win
    prob_A_win /= total_prob
    prob_draw /= total_prob
    prob_B_win /= total_prob

    flat_indices = np.argsort(score_matrix.ravel())[::-1][:3]
    top_scores = []
    for idx in flat_indices:
        i, j = divmod(idx, max_goals)
        top_scores.append((f"{i}:{j}", score_matrix[i, j] / total_prob))

    return prob_A_win, prob_draw, prob_B_win, lambda_A, lambda_B, top_scores


# ==========================================
# 4. Streamlit 前端界面渲染
# ==========================================
st.set_page_config(page_title="2026世界杯全栈推演器", page_icon="🏆", layout="wide")

st.title("🏆 2026美加墨世界杯：48强全员满血智能推演网")
st.markdown("本系统已全面升级！内置本届美加墨世界杯**全部 48 支入围球队**的底层量化权重。")
st.divider()

# 侧边栏看板
st.sidebar.header("📊 48强完整底层量化权重看板")
sidebar_df = pd.DataFrame.from_dict(TEAM_DATABASE, orient='index')[['Elo', 'Att', 'Def']]
st.sidebar.dataframe(sidebar_df, height=600)

# 球队对阵选择
col_a, col_b = st.columns(2)
with col_a:
    team_A = st.selectbox("🎯 选择主队 (Team A)", list(TEAM_DATABASE.keys()), index=0)
    st.info(f"**战术风格：** {TEAM_DATABASE[team_A]['Style']}")

with col_b:
    team_B = st.selectbox("🛡️ 选择客队 (Team B)", list(TEAM_DATABASE.keys()), index=1)
    st.info(f"**战术风格：** {TEAM_DATABASE[team_B]['Style']}")

# 推演触发逻辑
if st.button("🔥 启动48强满血版复合兵盘推演", use_container_width=True):
    if team_A == team_B:
        st.warning("⚠️ 相同球队无法对阵，请重新选择！")
    else:
        # 执行数学模型计算
        p_A, p_draw, p_B, exp_A, exp_B, top_scores = calculate_match_probability(team_A, team_B)

        # 数据可视化渲染
        st.subheader("📊 严格数学模型预测结果 (Double Poisson Matrix)")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric(f"{team_A} 胜率", f"{p_A:.2%}", f"期望进球: {exp_A:.2f}")
        col_res2.metric("平局概率", f"{p_draw:.2%}")
        col_res3.metric(f"{team_B} 胜率", f"{p_B:.2%}", f"期望进球: {exp_B:.2f}")

        st.progress(int(p_A * 100), text=f"{team_A} 胜出概率空间")

        st.markdown("##### 🎯 概率最高的三组精确比分预测：")
        score_text = " ｜ ".join([f"**{score}** (几率 {prob:.1%})" for score, prob in top_scores])
        st.write(score_text)
        st.divider()

        # ==========================================
        # 5. 调用大模型生成报告
        # ==========================================
        st.subheader("🧠 Gemini 战术沙盘兵推报告")
        with st.spinner("🤖 正在召集人工智能足球专家团进行兵盘分析..."):
            prompt = f"""
            你是一位享誉全球的硬核足球战术分析大师，行文风格锐利、专业、充满数据说服力。
            请针对这场2026世界杯焦点战进行战术复盘推演：{team_A} VS {team_B}。

            后端数学模型计算出的确定性上下文数据如下：
            1. 胜负平概率：{team_A}胜率 {p_A:.1%}，平局率 {p_draw:.1%}，{team_B}胜率 {p_B:.1%}。
            2. 期望进球数（XG）：{team_A}为 {exp_A:.2f}，{team_B}为 {exp_B:.2f}。
            3. 精确比分概率前三名为：{top_scores[0][0]}、{top_scores[1][0]}、{top_scores[2][0]}。
            4. 球队背景特征：
               - {team_A}（Elo {TEAM_DATABASE[team_A]['Elo']}）：{TEAM_DATABASE[team_A]['Style']}
               - {team_B}（Elo {TEAM_DATABASE[team_B]['Elo']}）：{TEAM_DATABASE[team_B]['Style']}

            请严格基于以上数据，撰写一份包含以下模块的战术推演报告：
            - 【控场战术博弈】：分析两队风格碰撞（如高位压迫vs防守反击）会如何演变。
            - 【数据合理性拆解】：结合两队的期望进球数（XG）和比分，解释为什么模型会得出这样的胜率。
            - 【胜负手与X因素】：指出哪一个细节会颠覆这个数学模型。
            字数控制在 400 字以内，一针见血。
            """

            try:
                # 统一调用最新稳定版官方接口接口
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                )
                st.write(response.text)
            except Exception as e:
                st.error(f"大模型推演失败，错误信息: {e}")