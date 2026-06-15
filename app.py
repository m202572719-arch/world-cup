import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
import google.genai as genai
import requests

# ==========================================
# 1. 初始化 Gemini 客户端
# ==========================================
try:
    client = genai.Client()
except Exception as e:
    st.error("Gemini API 客户端初始化失败，请检查云端 Secrets 配置。")

# ==========================================
# 2. 2026美加墨世界杯：官方正赛 48 强量化数据库（严格对照官方分组图）
# ==========================================
TEAM_DATABASE = {
    # --- Group A ---
    "墨西哥": {"Elo": 1820, "Att": 1.12, "Def": 0.90, "Pedigree": 1.05, "Alt_Fit": True, "Style": "东道主，中美洲技术流，阿兹特克高海拔魔鬼主场，脚下传切快。"},
    "南非": {"Elo": 1680, "Att": 0.96, "Def": 0.94, "Pedigree": 1.00, "Alt_Fit": False, "Style": "反击推进快，依靠整体就地小范围传导，但缺乏锋线强力终结者。"},
    "韩国": {"Elo": 1830, "Att": 1.15, "Def": 0.89, "Pedigree": 1.05, "Alt_Fit": False, "Style": "太极虎高位奔跑和体能极其疯狂，前场巨星闪光爆发力强。"},
    "捷克": {"Elo": 1795, "Att": 1.10, "Def": 0.90, "Pedigree": 1.00, "Alt_Fit": False, "Style": "典型欧洲身体对抗型，擅长高空球轰炸与两翼边路起球传中。"},

    # --- Group B ---
    "加拿大": {"Elo": 1790, "Att": 1.14, "Def": 0.93, "Pedigree": 1.00, "Alt_Fit": False, "Style": "东道主，两翼速度极快，纵深推进能力强，后防稍显年轻。"},
    "瑞士": {"Elo": 1880, "Att": 1.10, "Def": 0.84, "Pedigree": 1.05, "Alt_Fit": False, "Style": "战术执行力极高的硬骨头，整体链式防守非常严密，纪律性极强。"},
    "卡塔尔": {"Elo": 1715, "Att": 1.02, "Def": 0.95, "Pedigree": 1.00, "Alt_Fit": False, "Style": "亚洲杯常客，传控配合默契度高，极度依赖前场的反击突袭效率。"},
    "波黑": {"Elo": 1715, "Att": 1.02, "Def": 0.92, "Pedigree": 1.00, "Alt_Fit": False, "Style": "欧陆力量流派，身材高大，极其擅长定位球乱战与禁区头球砸门。"},

    # --- Group C ---
    "巴西": {"Elo": 2080, "Att": 1.38, "Def": 0.85, "Pedigree": 1.30, "Alt_Fit": True, "Style": "五星桑巴技术细腻，前场天才爆发力顶级，但近期后防单防有隐患。"},
    "摩洛哥": {"Elo": 1940, "Att": 1.20, "Def": 0.80, "Pedigree": 1.10, "Alt_Fit": False, "Style": "北非纯粹足球，退防密不透风，边路就地传切反击速度奇快。"},
    "海地": {"Elo": 1580, "Att": 0.92, "Def": 1.02, "Pedigree": 1.00, "Alt_Fit": False, "Style": "附加赛强悍黑马，球员爆发力和拼抢凶狠度强，但防守缺乏层次。"},
    "苏格兰": {"Elo": 1780, "Att": 1.04, "Def": 0.88, "Pedigree": 1.00, "Alt_Fit": False, "Style": "经典英伦硬朗作风，中场就地缠斗绞杀强，全队意志力极为坚韧。"},

    # --- Group D ---
    "美国": {"Elo": 1850, "Att": 1.15, "Def": 0.88, "Pedigree": 1.05, "Alt_Fit": False, "Style": "东道主，全留洋青年军，主场大球场冲击力和高频压迫侵略性极强。"},
    "巴拉圭": {"Elo": 1740, "Att": 0.92, "Def": 0.88, "Pedigree": 1.00, "Alt_Fit": True, "Style": "南美著名的低位硬骨头，死守坚固，球风极其极其凶悍凶狠。"},
    "澳大利亚": {"Elo": 1785, "Att": 1.04, "Def": 0.90, "Pedigree": 1.00, "Alt_Fit": False, "Style": "袋鼠军团身体强壮，高空争顶、定位球长传砸禁区是头号大杀器。"},
    "土耳其": {"Elo": 1845, "Att": 1.18, "Def": 0.88, "Pedigree": 1.00, "Alt_Fit": False, "Style": "星月军团作风彪悍，前场妖星单兵爆破力强，极其擅长打对攻乱战。"},

    # --- Group E ---
    "德国": {"Elo": 1980, "Att": 1.28, "Def": 0.88, "Pedigree": 1.25, "Alt_Fit": False, "Style": "战车强调中场控制与战术纪律，整体向前推进，阵地突击能力回升。"},
    "库拉索": {"Elo": 1550, "Att": 0.90, "Def": 1.05, "Pedigree": 1.00, "Alt_Fit": False, "Style": "北美神秘新面孔，多名海外归化坐镇，具备突出的单兵身体素质。"},
    "科特迪瓦": {"Elo": 1795, "Att": 1.14, "Def": 0.91, "Pedigree": 1.05, "Alt_Fit": False, "Style": "非洲大象身体素质爆表，中后场拦截硬度高，前场冲击力极强。"},
    "厄瓜多尔": {"Elo": 1870, "Att": 1.08, "Def": 0.83, "Pedigree": 1.00, "Alt_Fit": True, "Style": "高原跑不死体能怪，中场疯狗式就地逼抢，两翼边路插上飞快。"},

    # --- Group F ---
    "荷兰": {"Elo": 1950, "Att": 1.22, "Def": 0.79, "Pedigree": 1.15, "Alt_Fit": False, "Style": "顶级中卫群领衔防线，全攻全守底蕴，但进攻端缺乏绝对核心锋尖。"},
    "日本": {"Elo": 1925, "Att": 1.26, "Def": 0.84, "Pedigree": 1.05, "Alt_Fit": False, "Style": "亚洲地面传控天花板，全留洋阵容，高频就地反抢小组配合极其娴熟。"},
    "瑞典": {"Elo": 1855, "Att": 1.20, "Def": 0.88, "Pedigree": 1.10, "Alt_Fit": False, "Style": "北欧力量与技术的完美结合，锋线神锋终结力极高，攻防转换快。"},
    "突尼斯": {"Elo": 1760, "Att": 0.98, "Def": 0.88, "Pedigree": 1.00, "Alt_Fit": False, "Style": "北非纪律流，极度擅长低位摆大巴、中场长线死守绞杀，球风十分顽固。"},

    # --- Group G ---
    "比利时": {"Elo": 1905, "Att": 1.25, "Def": 0.89, "Pedigree": 1.05, "Alt_Fit": False, "Style": "欧洲红魔新老交替，进攻组织依旧犀利，但中后防线较怕速度冲击。"},
    "埃及": {"Elo": 1775, "Att": 1.12, "Def": 0.91, "Pedigree": 1.00, "Alt_Fit": False, "Style": "立足坚固防反，极端依赖前场核心巨星抓反击，抓失误一针见血。"},
    "伊朗": {"Elo": 1840, "Att": 1.12, "Def": 0.86, "Pedigree": 1.05, "Alt_Fit": False, "Style": "波斯铁骑，亚洲身体对抗对抗天花板，前场高塔抢点终结力极强。"},
    "新西兰": {"Elo": 1620, "Att": 0.94, "Def": 0.96, "Pedigree": 1.00, "Alt_Fit": False, "Style": "传统英式长传轰炸，身材高大高空对抗好，但后防脚下转身慢。"},

    # --- Group H ---
    "西班牙": {"Elo": 2045, "Att": 1.42, "Def": 0.82, "Pedigree": 1.20, "Alt_Fit": False, "Style": "极致地面传控配合，窒息的高位压迫，两侧年轻边锋爆破力极强。"},
    "佛得角": {"Elo": 1650, "Att": 0.95, "Def": 0.94, "Pedigree": 1.00, "Alt_Fit": False, "Style": "非洲技术型流派代表，战术极其灵活，长于就地防守反击。"},
    "沙特阿拉伯": {"Elo": 1695, "Att": 0.96, "Def": 0.96, "Pedigree": 1.00, "Alt_Fit": False, "Style": "小范围脚下技术出色，纪律性好，但在中东赛区以外客战战力打折。"},
    "乌拉圭": {"Elo": 1960, "Att": 1.30, "Def": 0.82, "Pedigree": 1.20, "Alt_Fit": True, "Style": "疯狗式中场强力绞杀与狂野高压反击结合，作风极其彪悍顽强。"},

    # --- Group I ---
    "法国": {"Elo": 2110, "Att": 1.52, "Def": 0.78, "Pedigree": 1.25, "Alt_Fit": False, "Style": "核武器级别的防守反击，两翼爆发速度恐怖，中场拦截硬度极高。"},
    "塞内加尔": {"Elo": 1865, "Att": 1.18, "Def": 0.85, "Pedigree": 1.05, "Alt_Fit": False, "Style": "特兰加雄狮，三线均有欧洲豪门核心，爆发力与对抗力量顶级。"},
    "伊拉克": {"Elo": 1670, "Att": 0.98, "Def": 0.97, "Pedigree": 1.00, "Alt_Fit": False, "Style": "中东强悍铁血球风，拼抢激烈对抗好，善于利用定位球高空抢点。"},
    "挪威": {"Elo": 1835, "Att": 1.24, "Def": 0.89, "Pedigree": 1.00, "Alt_Fit": False, "Style": "魔人神锋坐镇锋线，前场反击爆破和强力终结效率极其恐怖。"},

    # --- Group J ---
    "阿根廷": {"Elo": 2140, "Att": 1.45, "Def": 0.72, "Pedigree": 1.30, "Alt_Fit": False, "Style": "卫冕冠军，传控、逼抢与默契度完美，核心谢幕战精神属性拉满。"},
    "阿尔及利亚": {"Elo": 1810, "Att": 1.15, "Def": 0.90, "Pedigree": 1.00, "Alt_Fit": False, "Style": "北非地面传控派，脚下速率快，前场小范围地面渗透配合出色。"},
    "奥地利": {"Elo": 1835, "Att": 1.16, "Def": 0.87, "Pedigree": 1.00, "Alt_Fit": False, "Style": "极端高位运动量压迫，全员逼抢疯狂，攻防就地转换极其快。"},
    "约旦": {"Elo": 1690, "Att": 0.98, "Def": 0.95, "Pedigree": 1.00, "Alt_Fit": False, "Style": "亚洲坚韧硬骨头，全队退防阵型速度快，纪律性极好，擅长死守。"},

    # --- Group K ---
    "葡萄牙": {"Elo": 2010, "Att": 1.32, "Def": 0.84, "Pedigree": 1.10, "Alt_Fit": False, "Style": "三线球星云集极其豪华，反击推进速度奇快，单兵爆破终结力顶级。"},
    "民主刚果": {"Elo": 1675, "Att": 0.98, "Def": 0.96, "Pedigree": 1.00, "Alt_Fit": False, "Style": "纯力量对抗流派，防守拼抢极其凶狠，前场进攻主要依靠乱战。"},
    "乌兹别克斯坦": {"Elo": 1745, "Att": 1.02, "Def": 0.90, "Pedigree": 1.00, "Alt_Fit": False, "Style": "中亚白狼身体极强壮，战术纪律硬朗，中后场防守组织密不透风。"},
    "哥伦比亚": {"Elo": 1930, "Att": 1.24, "Def": 0.84, "Pedigree": 1.05, "Alt_Fit": True, "Style": "狂野南美传控流，细腻脚下与强悍对抗结合，近期状态极其火爆。"},

    # --- Group L ---
    "英格兰": {"Elo": 2050, "Att": 1.35, "Def": 0.80, "Pedigree": 1.15, "Alt_Fit": False, "Style": "总身价高昂，边路突破与阵地战传中轰炸能力顶级，作风严谨稳健。"},
    "克罗地亚": {"Elo": 1910, "Att": 1.12, "Def": 0.83, "Pedigree": 1.15, "Alt_Fit": False, "Style": "魔笛领衔格子军团，控节奏顶级，大赛心理与韧性极度恐怖。"},
    "加纳": {"Elo": 1720, "Att": 1.06, "Def": 0.98, "Pedigree": 1.00, "Alt_Fit": False, "Style": "阵中多名年轻英超妖星，身体天赋爆表，但后防线纪律容易散架。"},
    "巴拿马": {"Elo": 1710, "Att": 0.98, "Def": 0.92, "Pedigree": 1.00, "Alt_Fit": False, "Style": "中北美坚韧反击流，阵型退防层次极其紧凑，反击打纵深套路熟。"}
}

GLOBAL_AVG_GOALS = 1.35

# ==========================================
# 3. 自动化实时抓取功能 (ESPN 数据库数据源接口)
# ==========================================
def fetch_today_scores():
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            events = response.json().get("events", [])
            matches = []
            for event in events:
                status_text = event.get("status", {}).get("type", {}).get("detail", "")
                competitors = event.get("competitions", [{}])[0].get("competitors", [])
                home_team, away_team, home_score, away_score = "", "", "-", "-"
                for team in competitors:
                    t_name = team.get("team", {}).get("displayName", "")
                    score = team.get("score", "-")
                    if team.get("homeAway") == "home":
                        home_team = t_name
                        home_score = score
                    else:
                        away_team = t_name
                        away_score = score
                matches.append({"home": home_team, "away": away_team, "home_score": home_score, "away_score": away_score, "status": status_text})
            return matches
    except Exception as e:
        return None
    return []

# ==========================================
# 4. 融合精算数学引擎（去平滑 + 半全场 + 淘汰赛不含平局瓜分）
# ==========================================
def calculate_predictions_engine(team_A, team_B, venue_type, integrity_A, integrity_B, aggression_factor, is_knockout):
    data_A, data_B = TEAM_DATABASE[team_A], TEAM_DATABASE[team_B]
    att_A, def_A = data_A["Att"], data_A["Def"]
    att_B, def_B = data_B["Att"], data_B["Def"]
    
    # 动态主力残损降权
    att_A *= (integrity_A / 100.0)
    att_B *= (integrity_B / 100.0)
    
    # 全场基础泊松期望进球值计算
    lambda_A = att_A * def_B * GLOBAL_AVG_GOALS * aggression_factor
    lambda_B = att_B * def_A * GLOBAL_AVG_GOALS * aggression_factor
    
    # 注入三大东道主差异化主场优势
    if venue_type == "美国主场（NFL大型场馆 & 高分贝判罚优势）" and team_A == "美国":
        lambda_A *= 1.15
    elif venue_type == "加拿大主场（高纬度低温 & 人工合成快草皮）" and team_A == "加拿大":
        lambda_A *= 1.12
    elif venue_type == "印刷哥主场（2200米阿兹特克高原缺氧生态）":
        if team_A == "墨西哥": lambda_A *= 1.12
        if not data_A["Alt_Fit"]: lambda_A *= 0.92
        if not data_B["Alt_Fit"]: lambda_B *= 0.92

    # --- 🕒 半全场时间衰减分布计算 (上半场 43% / 下半场 57%) ---
    lambda_A_fh, lambda_B_fh = lambda_A * 0.43, lambda_B * 0.43
    lambda_A_sh, lambda_B_sh = lambda_A * 0.57, lambda_B * 0.57
    max_half_goals = 4
    
    matrix_fh = np.zeros((max_half_goals, max_half_goals))
    matrix_sh = np.zeros((max_half_goals, max_half_goals))
    for i in range(max_half_goals):
        for j in range(max_half_goals):
            matrix_fh[i, j] = poisson.pmf(i, lambda_A_fh) * poisson.pmf(j, lambda_B_fh)
            matrix_sh[i, j] = poisson.pmf(i, lambda_A_sh) * poisson.pmf(j, lambda_B_sh)
            
    fh_win = float(np.sum(np.tril(matrix_fh, -1)))
    fh_draw = float(np.sum(np.diag(matrix_fh)))
    fh_loss = float(np.sum(np.triu(matrix_fh, 1)))
    sh_win = float(np.sum(np.tril(matrix_sh, -1)))
    sh_draw = float(np.sum(np.diag(matrix_sh)))
    sh_loss = float(np.sum(np.triu(matrix_sh, 1)))

    ht_ft_space = {
        "胜-胜": fh_win * sh_win, "胜-平": fh_win * sh_draw, "胜-负": fh_win * sh_loss,
        "平-胜": fh_draw * sh_win, "平-平": fh_draw * sh_draw, "平-负": fh_draw * sh_loss,
        "负-胜": fh_loss * sh_win, "负-平": fh_loss * sh_draw, "负-负": fh_loss * j * sh_loss
    }
    total_ht_ft = sum(ht_ft_space.values()) + 1e-6
    for k in ht_ft_space: ht_ft_space[k] /= total_ht_ft
    top_ht_ft = sorted(ht_ft_space.items(), key=lambda x: x[1], reverse=True)[:3]

    # --- 📊 单场 90 分钟去平滑回归全场比分精算 ---
    max_fg = 6
    score_matrix = np.zeros((max_fg, max_fg))
    for i in range(max_fg):
        for j in range(max_fg):
            score_matrix[i, j] = poisson.pmf(i, lambda_A) * poisson.pmf(j, lambda_B)
            
    if aggression_factor > 1.1:
        score_matrix[0, 0] *= 0.75
        score_matrix[1, 1] *= 0.80
        score_matrix[1, 0] *= 0.85
        score_matrix[0, 1] *= 0.85
        score_matrix /= (np.sum(score_matrix) + 1e-6)

    p_win_A_raw = float(np.sum(np.tril(score_matrix, -1)))
    p_draw_raw = float(np.sum(np.diag(score_matrix)))
    p_win_B_raw = float(np.sum(np.triu(score_matrix, 1)))
    
    # 融入条件概率分配逻辑与淘汰赛底蕴加权修正
    if is_knockout:
        base_w_A = p_win_A_raw / (p_win_A_raw + p_win_B_raw + 1e-6)
        base_w_B = p_win_B_raw / (p_win_A_raw + p_win_B_raw + 1e-6)
        pedigree_gap = data_A["Pedigree"] - data_B["Pedigree"]
        
        weight_A = base_w_A * (1.0 + pedigree_gap * 0.1)
        weight_B = base_w_B * (1.0 - pedigree_gap * 0.1)
        total_w = weight_A + weight_B + 1e-6
        
        p_win_A = p_win_A_raw + p_draw_raw * (weight_A / total_w)
        p_win_B = p_win_B_raw + p_draw_raw * (weight_B / total_w)
        p_draw = 0.0
    else:
        # 常规联赛制：结合冠军底蕴进行盘口抗跌微调
        pedigree_gap = data_A["Pedigree"] - data_B["Pedigree"]
        p_win_A = p_win_A_raw + (pedigree_gap * 0.05)
        p_win_B = p_win_B_raw - (pedigree_gap * 0.05)
        p_draw = p_draw_raw
        total_res = p_win_A + p_draw + p_win_B
        p_win_A, p_draw, p_win_B = p_win_A/total_res, p_draw/total_res, p_win_B/total_res

    flat_indices = np.argsort(score_matrix.ravel())[::-1][:3]
    top_scores = []
    for idx in flat_indices:
        i, j = divmod(idx, max_fg)
        top_scores.append((f"{i}:{j}", score_matrix[i, j] / np.sum(score_matrix)))
        
    return p_win_A, p_draw, p_win_B, lambda_A, lambda_B, top_scores, top_ht_ft

# ==========================================
# 5. Streamlit 高栈渲染控制台
# ==========================================
st.set_page_config(page_title="2026世界杯精密推演系统", page_icon="🏆", layout="wide")

st.title("🏆 2026美加墨世界杯：48强正赛官方数据高级精算与全维度辅助控制台")
st.markdown("⚠️ **生产级生产环境闭环版：** 深度融合 ESPN 全球比分同步机制、独立时间衰减 HT/FT 矩阵及淘汰赛条件概率瓜分逻辑。")
st.divider()

# 侧边栏：同步 ESPN 实时果组件
with st.sidebar:
    st.header("🔄 自动化实时赛果同步中心")
    if st.button("一键同步今日最新真实比分"):
        with st.spinner("正在从 ESPN 全球数据库拉取赛果..."):
            scores = fetch_today_scores()
            if scores:
                st.success(f"成功同步 {len(scores)} 场最新比赛！")
                for match in scores:
                    st.write(f"**{match['status']}**")
                    st.write(f"{match['home']} {match['home_score']} : {match['away_score']} {match['away']}")
                    st.write("---")
            else:
                st.warning("暂无正在进行或已结束的赛事，或接口超时。")

# 主界面：精算预测表单
col_ctl1, col_ctl2 = st.columns(2)

with col_ctl1:
    st.subheader("📋 赛事基本面选择")
    team_list = list(TEAM_DATABASE.keys())
    team_A = st.selectbox("🎯 选择主队 (Team A)", team_list, index=4)  # 默认加拿大
    team_B = st.selectbox("🛡️ 选择客队 (Team B)", team_list, index=7)  # 默认波黑
    is_knockout = st.checkbox("🏆 开启淘汰赛赛制 (自动通过条件概率瓜分平局率，精算独赢晋级率)")

with col_ctl2:
    st.subheader("⚙️ 彩票风控变数动态微调")
    venue = st.radio(
        " Stadium 设定本场赛地的地缘物理环境因子",
        ["中立场地 / 其他常规赛区", "美国主场（NFL大型场馆 & 高分贝判罚优势）", "加拿大主场（高纬度低温 & 人工合成快草皮）", "墨西哥主场（2200米阿兹特克高原缺氧生态）"],
        index=2
    )
    agg_factor = st.slider("🔥 战术博弈激进烈度（强行压制低平比分，拉大波胆方差）", 0.8, 1.8, 1.3, step=0.1)

st.markdown("##### 🩺 临场两队核心伤情/大轮换折损")
col_inj1, col_inj2 = st.columns(2)
with col_inj1: integrity_A = st.slider(f"【{team_A}】阵容战力完整度 (%)", 50, 100, 100)
with col_inj2: integrity_B = st.slider(f"【{team_B}】阵容战力完整度 (%)", 50, 100, 100)

st.divider()

st.info(f"💡 **主队基本面分析 ({team_A})：** {TEAM_DATABASE[team_A]['Style']}")
st.info(f"💡 **客队基本面分析 ({team_B})：** {TEAM_DATABASE[team_B]['Style']}")

if st.button("🔥 启动多维泊松时间矩阵进行足彩精密兵盘推演", use_container_width=True):
    p_A, p_draw, p_B, exp_A, exp_B, top_scores, top_ht_ft = calculate_advanced_match(
        team_A, team_B, venue, integrity_A, integrity_B, agg_factor, is_knockout
    )
    
    # 渲染全场精算
    st.subheader("📊 独家足彩胜平负、全场比分精算期望")
    res_1, res_2, res_3 = st.columns(3)
    res_1.metric(f"【胜】{team_A} 胜出率", f"{p_A:.2%}", f"修正后期望进球: {exp_A:.2f}")
    if is_knockout:
        res_2.metric("【平】平局概率", "已按底蕴加时条件瓜分", delta="淘汰赛制制止平局")
    else:
        res_2.metric("【平】平局概率", f"{p_draw:.2%}")
    res_3.metric(f"【负】{team_B} 胜出率", f"{p_B:.2%}", f"修正后期望进球: {exp_B:.2f}")
    
    st.progress(int(p_A * 100), text=f"{team_A} 独赢全场概率分布")
    
    st.markdown("##### 🎯 全场精确波胆（比分）几率前三预测（Dixon-Coles 抑制）：")
    score_text = " ｜ ".join([f"预测 **{score}** (精确几率 {prob:.1%})" for score, prob in top_scores])
    st.write(score_text)
    st.divider()
    
    # 渲染半全场精算看板
    st.subheader("⏳ 全网独家首发：半全场（HT/FT）高赔率冷门几率精算")
    ht_col1, ht_ft_col2, ht_ft_col3 = st.columns(3)
    ht_col1.metric("🔥 黄金选项 1", f"【{top_ht_ft[0][0]}】", f"精确几率: {top_ht_ft[0][1]:.2%}")
    ht_ft_col2.metric("🎯 次热防线 2", f"【{top_ht_ft[1][0]}】", f"精确几率: {top_ht_ft[1][1]:.2%}")
    ht_ft_col3.metric("🔮 冷门博弈 3", f"【{top_ht_ft[2][0]}】", f"精确几率: {top_ht_ft[2][1]:.2%}")
    st.divider()
    
    # 调用全新标准的正式版接口，吐出足彩下注报告
    st.subheader("🧠 Gemini 工业级足彩战术博弈深度内参")
    with st.spinner("🤖 正在安全调度生产级稳定内核结合半全场九项进行决策分析..."):
        pedigree_A = TEAM_DATABASE[team_A]["Pedigree"]
        pedigree_B = TEAM_DATABASE[team_B]["Pedigree"]
        
        prompt = f"""
        你是一位享誉全球的硬核足球足彩精算大师，行文风格锐利、专业、极具博弈论视角的金钱说服力。
        请针对这场2026世界杯焦点大战进行足彩下注层面的战术推演：{team_A} VS {team_B}。
        
        后端双泊松精算模型给出的确定性上下文如下：
        1. 全场胜率：{team_A}胜率 {p_A:.1%}，平局率 {p_draw:.1%}，{team_B}胜率 {p_B:.1%}（淘汰赛机制已生效={is_knockout}）。
        2. 期望进球（XG）：{team_A}为 {exp_A:.2f}，{team_B}为 {exp_B:.2f}。
        3. 精算半全场概率前三名为：{top_ht_ft[0][0]}({top_ht_ft[0][1]:.1%})、{top_ht_ft[1][0]}({top_ht_ft[1][1]:.1%})、{top_ht_ft[2][0]}({top_ht_ft[2][1]:.1%})。
        4. 选定赛场变量区域：{venue}。
        5. 全场比分概率前三名：{top_scores[0][0]}、{top_scores[1][0]}。
        6. 战术本底：
           - {team_A}（底蕴权重 {pedigree_A}）：{TEAM_DATABASE[team_A]['Style']}
           - {team_B}（底蕴权重 {pedigree_B}）：{TEAM_DATABASE[team_B]['Style']}
        
        请结合时间衰减加成、上半场阵地大巴和下半场体能博弈特征，撰写一份包含以下模块的足彩策略：
        - 【半全场走势拆解】：深度剖析为什么模型会得出前三名的【半全场组合】（例如分析为什么在当前物理环境下容易出现平-胜）。
        - 【足彩总进球与大小球配单推荐】：结合去平滑期望值与半全场组合，明确给出【大小球盘口】与【半全场高胜率串关配单】斩钉截铁的下注策略。
        - 【足彩X因素防范】：直接指出哪些临场突发变数会颠覆这个冷冰冰的数学模型。
        字数控制在 400 字以内，一针见血。
        """
        try:
            # 严格调度新版 SDK 通用正式内核 'gemini-2.5-flash'，彻底阻断 403/404 权限黑洞
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            st.write(response.text)
        except Exception as e:
            st.error(f"大模型内参生成失败，错误信息: {e}")
