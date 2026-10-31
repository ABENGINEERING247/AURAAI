import streamlit as st
import os, json, requests
from datetime import datetime
from io import BytesIO
from openai import OpenAI
import pandas as pd

st.set_page_config(page_title="AURA | Cricket & Football Intelligence", page_icon="🏆", layout="wide")

# Secrets are preferred; environment variables also work for local deployment.
def secret(name, default=""):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

OPENAI_API_KEY = secret("OPENAI_API_KEY")
OPENAI_MODEL = secret("OPENAI_MODEL", "gpt-4o-mini")
FOOTBALL_API_KEY = secret("API_FOOTBALL_KEY")
CRICKET_API_KEY = secret("CRICAPI_KEY")
TWITCH_CLIENT_ID = secret("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET = secret("TWITCH_CLIENT_SECRET")

st.markdown("""
<style>
.stApp {background:linear-gradient(135deg,#07111f 0%,#0b1d32 55%,#10283b 100%);color:#f4f7fb}
h1,h2,h3 {color:#f5f8ff}
[data-testid="stMetric"]{background:#10263c;padding:16px;border:1px solid #24445f;border-radius:14px}
.aura-sub{color:#a8c2d8}

/* Force Navigation Tab Text to Pure White */
div[data-testid="stTabs"] button {
    font-weight: 700 !important;
    color: #ffffff !important;
}

/* Selected Tab Color Highlight */
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #4da6ff !important;
}

/* Sidebar Text Color Fix */
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🏆 AURA SPORTS INTELLIGENCE")
st.markdown("<p class='aura-sub'>Your AI companion for Cricket, Football, player analysis, live data and professional reports.</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ AURA Control Center")
    st.caption("Securely add keys in Streamlit → App settings → Secrets.")
    st.success("OpenAI: Connected" if OPENAI_API_KEY else "OpenAI: Demo mode — add OPENAI_API_KEY")
    st.info("Live sports feeds require provider keys. Without them, AURA clearly labels sample data.")
    sport = st.selectbox("Primary sport", ["Cricket", "Football", "Both"])
    st.divider()
    st.markdown("**AURA modules**")
    st.write("• AI Sports Assistant")
    st.write("• Player rankings & statistics")
    st.write("• Match analysis")
    st.write("• Tabular reports")
    st.write("• PDF / DOCX export")
    st.write("• Twitch community mode")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "report" not in st.session_state:
    st.session_state.report = pd.DataFrame(columns=["Category","Metric","Value","Notes"])

def ask_aura(prompt, selected_sport):
    if not OPENAI_API_KEY:
        return ("AURA is running in demo mode. Add OPENAI_API_KEY to Streamlit Secrets for AI-powered answers. "
                "I can help with rules, formats, tactics, player comparisons, training plans, match previews, "
                "scorecard interpretation and report structure. For live results or rankings, connect a sports data provider.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    system = f"""You are AURA, a knowledgeable cricket and football analyst and assistant.
Focus: {selected_sport}. Answer accurately and in a professional structure.
Cover rules, formats, history, tactics, teams, player profiles, statistics, training, match analysis,
fantasy/community engagement and explanations. Never invent live scores, rankings, injuries or transfers.
If real-time data is not supplied, state that it must be verified from a live data provider.
Use headings, concise tables when useful, and explain statistical context."""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":system}] + st.session_state.messages[-10:] + [{"role":"user","content":prompt}],
        temperature=0.35
    )
    return response.choices[0].message.content

def football_players():
    if not FOOTBALL_API_KEY:
        return None, "No API_FOOTBALL_KEY configured. Showing sample ranking."
    try:
        league = st.session_state.get("league_id", 39)
        season = st.session_state.get("season", 2025)
        url="https://v3.football.api-sports.io/players"
        r=requests.get(url, headers={"x-apisports-key":FOOTBALL_API_KEY},
                       params={"league":league,"season":season}, timeout=20)
        r.raise_for_status()
        data=r.json().get("response",[])
        rows=[]
        for item in data:
            p=item.get("player",{}); stats=(item.get("statistics") or [{}])[0]
            goals=(stats.get("goals") or {}).get("total") or 0
            assists=(stats.get("goals") or {}).get("assists") or 0
            rows.append({"Rank":0,"Player":p.get("name","Unknown"),"Team":(stats.get("team") or {}).get("name",""),
                         "Goals":goals,"Assists":assists,"Appearances":(stats.get("games") or {}).get("appearences",0)})
        df=pd.DataFrame(rows)
        if not df.empty:
            df=df.sort_values(["Goals","Assists"],ascending=False).head(20).reset_index(drop=True)
            df.insert(0,"Rank",range(1,len(df)+1))
        return df, "API-Sports live player statistics"
    except Exception as e:
        return None, f"Football provider error: {e}"

def cricket_players():
    if not CRICKET_API_KEY:
        return None, "No CRICAPI_KEY configured. Showing sample ranking."
    try:
        url="https://api.cricapi.com/v1/players"
        r=requests.get(url,params={"apikey":CRICKET_API_KEY,"offset":0},timeout=20)
        r.raise_for_status()
        payload=r.json()
        rows=[]
        for i,p in enumerate(payload.get("data",[])[:20],1):
            rows.append({"Rank":i,"Player":p.get("name","Unknown"),"Country":p.get("country",""),
                         "Role":p.get("role","Not supplied"),"Ranking metric":"Provider player directory"})
        return pd.DataFrame(rows), "CricAPI player directory (not a performance ranking)"
    except Exception as e:
        return None, f"Cricket provider error: {e}"

def sample_rankings(which):
    if which=="Football":
        return pd.DataFrame([
            {"Rank":1,"Player":"Sample Player A","Team":"Sample FC","Goals":28,"Assists":9,"Appearances":32},
            {"Rank":2,"Player":"Sample Player B","Team":"Sample United","Goals":24,"Assists":12,"Appearances":30},
            {"Rank":3,"Player":"Sample Player C","Team":"Sample City","Goals":21,"Assists":7,"Appearances":29}])
    return pd.DataFrame([
        {"Rank":1,"Player":"Sample Batter A","Country":"Sample Country","Role":"Batter","Ranking metric":"Demo only"},
        {"Rank":2,"Player":"Sample All-rounder B","Country":"Sample Country","Role":"All-rounder","Ranking metric":"Demo only"},
        {"Rank":3,"Player":"Sample Bowler C","Country":"Sample Country","Role":"Bowler","Ranking metric":"Demo only"}])

tabs=st.tabs(["🏠 Overview","💬 AURA Chat","📊 Player Rankings","📋 Reports & Export","🎮 Twitch Community","📚 Sports Knowledge"])

with tabs[0]:
    st.subheader("One hub. Two global sports.")
    c1,c2,c3=st.columns(3)
    c1.metric("Sports covered","Cricket + Football")
    c2.metric("Report formats","PDF + DOCX")
    c3.metric("AI engine", "OpenAI" if OPENAI_API_KEY else "Demo")
    st.markdown("### What AURA can do")
    cols=st.columns(2)
    with cols[0]:
        st.markdown("**🏏 Cricket**")
        st.write("Test, ODI, T20, T10, domestic leagues, batting, bowling, fielding, scorecards, rules, records, player roles, tactics and match reports.")
    with cols[1]:
        st.markdown("**⚽ Football**")
        st.write("Club and international football, leagues, tournaments, formations, tactics, player profiles, goals, assists, fixtures, standings and match analysis.")
    st.warning("Live scores, player rankings and standings are only live when a supported data provider is connected. AURA does not label demo data as live.")

with tabs[1]:
    st.subheader("Chat with AURA")
    st.caption("Ask in natural language. Request a match preview, player comparison, rules explanation or structured report.")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    prompt=st.chat_input("Ask AURA about cricket or football…")
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("AURA is thinking…"):
                answer=ask_aura(prompt,sport)
                st.markdown(answer)
        st.session_state.messages.append({"role":"assistant","content":answer})

with tabs[2]:
    st.subheader("Player rankings & performance")
    ranking_sport=st.radio("Choose sport",["Cricket","Football"],horizontal=True)
    if ranking_sport=="Football":
        a,b=st.columns(2)
        with a: st.number_input("League ID (API-Sports)",min_value=1,value=39,key="league_id")
        with b: st.number_input("Season",min_value=2000,max_value=2035,value=2025,key="season")
        if st.button("Refresh football player data"):
            st.session_state.football_refresh=True
        df,msg=football_players()
    else:
        if st.button("Refresh cricket player data"):
            st.session_state.cricket_refresh=True
        df,msg=cricket_players()
    if df is None or df.empty:
        df=sample_rankings(ranking_sport)
        st.info(msg+" The table below is illustrative sample data, not a current official ranking.")
    else:
        st.success(msg)
        if ranking_sport=="Cricket":
            st.warning("CricAPI player directory is not an official ICC batting/bowling/all-rounder ranking. Connect an authorized rankings feed to display official rank positions.")
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.download_button("Download ranking CSV",df.to_csv(index=False).encode("utf-8"),file_name=f"aura_{ranking_sport.lower()}_players.csv",mime="text/csv")

with tabs[3]:
    st.subheader("Professional sports report builder")
    with st.form("report_form"):
        category=st.selectbox("Report type",["Match Analysis","Player Performance","Team Comparison","Tournament Summary","Training Plan","Scouting Report"])
        title=st.text_input("Report title",placeholder="e.g. Match analysis — Team A vs Team B")
        metric=st.text_input("Key metric / section",placeholder="e.g. Possession, strike rate, wickets")
        value=st.text_input("Finding / value")
        notes=st.text_area("Analysis / notes")
        submitted=st.form_submit_button("Add row to report")
    if submitted:
        row=pd.DataFrame([{"Category":category,"Metric":metric,"Value":value,"Notes":notes}])
        st.session_state.report=pd.concat([st.session_state.report,row],ignore_index=True)
        if title: st.session_state.report_title=title
    st.markdown("### Report table")
    st.dataframe(st.session_state.report,use_container_width=True,hide_index=True)
    if st.button("Clear report"):
        st.session_state.report=pd.DataFrame(columns=["Category","Metric","Value","Notes"])
        st.rerun()
    report_title=st.session_state.get("report_title","AURA Sports Report")
    csv=st.session_state.report.to_csv(index=False).encode("utf-8")
    st.download_button("Download Excel-compatible CSV",csv,"aura_sports_report.csv","text/csv")
    if st.button("Generate PDF and DOCX"):
        if st.session_state.report.empty:
            st.warning("Add at least one report row first.")
        else:
            try:
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import A4, landscape
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                from reportlab.lib.styles import getSampleStyleSheet
                from docx import Document
                pdf_buffer=BytesIO()
                doc=SimpleDocTemplate(pdf_buffer,pagesize=landscape(A4))
                styles=getSampleStyleSheet()
                story=[Paragraph(report_title,styles["Title"]),Paragraph("Generated by AURA Sports Intelligence",styles["Normal"]),Spacer(1,14)]
                data=[list(st.session_state.report.columns)]+st.session_state.report.fillna("").astype(str).values.tolist()
                table=Table(data,repeatRows=1)
                table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#12304a")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.5,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#edf3f8")])]))
                story.append(table); doc.build(story)
                pdf_buffer.seek(0)
                docx_buffer=BytesIO()
                word=Document(); word.add_heading(report_title,0); word.add_paragraph("Generated by AURA Sports Intelligence")
                word.add_heading("Report findings",level=1)
                t=word.add_table(rows=1,cols=len(st.session_state.report.columns)); t.style="Light Shading Accent 1"
                for i,col in enumerate(st.session_state.report.columns): t.rows[0].cells[i].text=str(col)
                for _,r in st.session_state.report.iterrows():
                    cells=t.add_row().cells
                    for i,col in enumerate(st.session_state.report.columns): cells[i].text=str(r[col])
                word.save(docx_buffer); docx_buffer.seek(0)
                st.download_button("⬇️ Download PDF",pdf_buffer.getvalue(),"AURA_Sports_Report.pdf","application/pdf")
                st.download_button("⬇️ Download DOCX",docx_buffer.getvalue(),"AURA_Sports_Report.docx","application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            except Exception as e: st.error(f"Export error: {e}")

with tabs[4]:
    st.subheader("Twitch community mode")
    st.write("AURA can be extended into a Twitch chat companion: viewers can request sports trivia, predict match outcomes, compare players and trigger streamer-friendly summaries.")
    st.markdown("**Suggested chat commands**")
    st.code("""!aura <question>       Ask AURA a cricket/football question
!cricketquiz           Start a cricket trivia question
!footballquiz          Start a football trivia question
!compare <player1> <player2>
!matchreport           Show the latest prepared match report
!rules <topic>         Explain a rule""")
    st.caption("Twitch credentials are reserved for a separate Twitch bot/extension integration. Never put Twitch secrets in public GitHub code.")
    st.info("For the hackathon MVP, use this Streamlit dashboard as the streamer control panel, then connect a TwitchIO bot or Twitch EventSub/IRC integration as a separate service.")

with tabs[5]:
    st.subheader("Sports knowledge center")
    k1,k2=st.columns(2)
    with k1:
        st.markdown("**Cricket coverage**")
        st.write("Formats and laws • innings and overs • batting/bowling techniques • fielding positions • DLS overview • powerplays • scorecards • player roles • domestic and international competitions.")
    with k2:
        st.markdown("**Football coverage**")
        st.write("Laws of the game • positions • formations • pressing and build-up • offside • set pieces • match statistics • club and national competitions • player performance metrics.")
    st.markdown("**Example prompts**")
    st.write("- Compare two players using recent verified statistics.")
    st.write("- Explain the difference between Test, ODI and T20 cricket.")
    st.write("- Analyse a 4-3-3 football formation against a low block.")
    st.write("- Create a structured post-match report with a statistics table.")
    st.write("- Build a scouting report for a striker / fast bowler.")
st.divider()
st.caption("AURA Sports Intelligence • Streamlit application • Verify live sports data against the official competition/provider source.")
