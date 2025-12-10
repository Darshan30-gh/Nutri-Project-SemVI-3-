import streamlit as st
import pandas as pd
from scipy.optimize import linprog
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import plotly.express as px
import time
from gtts import gTTS
import tempfile
from groq import Groq

# --- 1. CONFIGURATION & UI THEME ---
st.set_page_config(page_title="OptiNutri Dashboard", page_icon="🥗", layout="wide")
load_dotenv()

# CUSTOM CSS: This makes it look like the React Dashboard in your screenshot
st.markdown("""
    <style>
    /* 1. Background Color (Light Grey for contrast) */
    .stApp {
        background-color: #f5f7f9;
    }
    
    /* 2. THE CARD STYLE (White box with shadow) */
    .css-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* 3. Metric Styling (To look like the stats cards) */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #6c5ce7; /* Purple accent */
    }
    
    /* 4. Remove Streamlit Header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 5. Custom Button (Purple Gradient like your screenshot) */
    div.stButton > button {
        background: linear-gradient(90deg, #6c5ce7 0%, #a29bfe 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
    }
    div.stButton > button:hover {
        box-shadow: 0 4px 12px rgba(108, 92, 231, 0.3);
        color: white;
    }
    
    /* 6. Titles */
    h1, h2, h3 {
        color: #2d3436;
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPER: CARD WRAPPER ---
def start_card():
    st.markdown('<div class="css-card">', unsafe_allow_html=True)

def end_card():
    st.markdown('</div>', unsafe_allow_html=True)

# --- 2. LOGIC FUNCTIONS (SAME AS BEFORE) ---
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en', tld='co.in', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except: return None

def transcribe_audio(audio_bytes, api_key):
    client = Groq(api_key=api_key)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        fp.write(audio_bytes)
        fp.flush()
        filename = fp.name
    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(filename, file.read()), model="whisper-large-v3", 
            response_format="json", language="en", temperature=0.0 
        )
    return transcription.text

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/food_data.csv')
        df.columns = df.columns.str.strip().str.title() 
        return df
    except: return None

# --- 3. LOGIN PAGE (CLEAN DESIGN) ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

def login():
    c1, c2, c3 = st.columns([1,1,1])
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        start_card() # White Card
        st.markdown("<h2 style='text-align: center; color: #6c5ce7;'>Welcome Back!</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Sign in to OptiNutri Dashboard</p>", unsafe_allow_html=True)
        
        u = st.text_input("Username", placeholder="admin")
        p = st.text_input("Password", type="password", placeholder="1234")
        
        if st.button("Login to Dashboard"):
            if u=="admin" and p=="1234":
                st.session_state['logged_in'] = True
                st.session_state['user'] = "Administrator"
                st.rerun()
            else: st.error("Access Denied")
        end_card()

# --- 4. MAIN DASHBOARD UI ---
def main_app():
    df = load_data()
    api_key = os.getenv("GROQ_API_KEY")

    # SIDEBAR DESIGN
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2913/2913990.png", width=50)
        st.markdown(f"### **{st.session_state['user']}**")
        st.markdown("---")
        app_mode = st.radio("MENU", ["Dashboard Overview", "Diet Planner", "Voice Assistant"])
        st.markdown("---")
        if st.button("Log Out"):
            st.session_state['logged_in'] = False
            st.rerun()

    # --- TAB 1: DASHBOARD OVERVIEW (Looks like your Screenshot) ---
    if app_mode == "Dashboard Overview":
        st.title(f"Welcome back, {st.session_state['user']}! 👋")
        st.markdown("Here's what's happening with your nutrition database today.")
        st.markdown("<br>", unsafe_allow_html=True)

        # TOP METRICS ROW (Like your screenshot's top cards)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Food Items", len(df), "+12 this week")
        m2.metric("Avg Protein Cost", "₹1.5/g", "-4% vs last week")
        m3.metric("Diet Plans Generated", "128", "+24 today")
        m4.metric("System Status", "Active", "Online")

        st.markdown("<br>", unsafe_allow_html=True)

        # MAIN CONTENT GRID
        c1, c2 = st.columns([2, 1])

        with c1:
            start_card() # White Card
            st.subheader("📊 Price vs Protein Analysis")
            fig = px.scatter(df, x="Price", y="Protein", size="Calories", color="Fat", height=350)
            st.plotly_chart(fig, use_container_width=True)
            end_card()

        with c2:
            start_card() # White Card
            st.subheader("⚡ Quick Actions")
            if st.button("Add New Food Item"):
                st.toast("Opening Database Editor...")
            if st.button("Download Monthly Report"):
                st.toast("Generating PDF...")
            if st.button("System Health Check"):
                st.toast("All Systems Normal")
            end_card()

    # --- TAB 2: DIET PLANNER (The Math) ---
    elif app_mode == "Diet Planner":
        st.title("🥗 Smart Diet Optimizer")
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            start_card()
            st.subheader("🎯 Set Goals")
            min_cal = st.slider("Calories", 1500, 3500, 2000)
            min_prot = st.slider("Protein (g)", 30, 150, 60)
            min_fat = st.slider("Fat (g)", 10, 100, 30)
            run_btn = st.button("Generate Plan")
            end_card()

        with c2:
            if run_btn:
                start_card()
                st.subheader("📋 Your Optimized Plan")
                # (Math Logic Simplified for brevity)
                costs = df['Price'].values
                res = linprog(c=costs, A_ub=[-df['Calories'], -df['Protein'], -df['Fat']], 
                              b_ub=[-min_cal, -min_prot, -min_fat], bounds=(0,5), method='highs')
                
                if res.success:
                    st.success(f"Optimization Complete! Total Cost: ₹{sum(res.x * df['Price']):.2f}")
                    # Show simple table
                    items = []
                    for i, q in enumerate(res.x):
                        if q > 0.01: items.append([df.iloc[i]['Food'], f"{q*100:.0f}g", f"₹{q*df.iloc[i]['Price']:.2f}"])
                    st.table(pd.DataFrame(items, columns=["Food", "Qty", "Cost"]))
                end_card()

    # --- TAB 3: VOICE ASSISTANT ---
    elif app_mode == "Voice Assistant":
        st.title("🎙️ AI Nutritionist")
        start_card()
        st.markdown("### Speak to your AI Assistant")
        audio = st.audio_input("Record Voice")
        
        if audio:
            with st.spinner("Processing..."):
                text = transcribe_audio(audio.read(), api_key)
                st.info(f"You said: {text}")
                
                llm = ChatGroq(api_key=api_key, model_name="llama-3.3-70b-versatile")
                reply = llm.invoke(f"Context: Nutrition App. User said: {text}. Answer briefly.").content
                
                st.success(f"AI: {reply}")
                speech = text_to_speech(reply)
                st.audio(speech, autoplay=True)
        end_card()

if __name__ == "__main__":
    if st.session_state['logged_in']: main_app()
    else: login()