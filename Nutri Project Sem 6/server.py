from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from scipy.optimize import linprog
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from groq import Groq
from gtts import gTTS
import tempfile
import base64
from fpdf import FPDF

app = Flask(__name__)
CORS(app)
load_dotenv()

# --- 1. LOAD DATA ---
try:
    df = pd.read_csv('data/food_data.csv')
    df.columns = df.columns.str.strip().str.title()
    # Clean numeric data
    for col in ['Price', 'Protein', 'Calories', 'Fat']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    print(f"✅ Data Loaded: {len(df)} items")
except Exception as e:
    print(f"❌ Data Error: {e}")
    df = None

# --- 2. OPTIMIZATION ENGINE (WITH DIET FILTER) ---
@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    try:
        min_cal = float(data.get('calories', 1800))
        min_prot = float(data.get('protein', 50))
        diet_type = data.get('diet_type', 'both').lower() # New Parameter
        
        if df is None: return jsonify({"success": False, "message": "Database not loaded"}), 500

        # --- A. FILTER DATA BASED ON DIET TYPE ---
        working_df = df.copy()
        
        if diet_type == 'veg':
            # Keywords to EXCLUDE for Vegetarians
            non_veg_keywords = ['Chicken', 'Egg', 'Fish', 'Mutton', 'Beef', 'Prawn', 'Lamb', 'Pork', 'Ham', 'Bacon', 'Salami']
            pattern = '|'.join(non_veg_keywords)
            # Keep rows that do NOT match the non-veg keywords
            working_df = working_df[~working_df['Food'].str.contains(pattern, case=False, na=False)]
        
        if len(working_df) == 0:
            return jsonify({"success": False, "message": "No food items left after filtering!"})

        # --- B. RUN OPTIMIZATION ---
        costs = working_df['Price'].values
        A_ub = [-working_df['Calories'].values, -working_df['Protein'].values, -working_df['Fat'].values]
        b_ub = [-min_cal, -min_prot, -15] 
        bounds = [(0, 3) for _ in range(len(working_df))] # Max 3 units per item

        res = linprog(c=costs, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            result_items = []
            total_cost = 0
            for i, qty in enumerate(res.x):
                if qty > 0.05:
                    item = working_df.iloc[i]
                    cost = qty * item['Price']
                    result_items.append({
                        "food": item['Food'],
                        "qty": f"{qty*100:.0f}g",
                        "cost": f"₹{cost:.2f}"
                    })
                    total_cost += cost
            
            return jsonify({
                "success": True, 
                "total_cost": f"₹{total_cost:.2f}", 
                "plan": result_items
            })
        else:
            return jsonify({"success": False, "message": "Targets too high for this diet. Try lowering Protein."})
            
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# --- 3. ANALYTICS (Sorted Top 15) ---
@app.route('/analytics-data', methods=['GET'])
def analytics():
    if df is None: return jsonify({"error": "No data"}), 500
    top_protein = df.sort_values(by='Protein', ascending=False).head(15)
    return jsonify({
        "total_items": len(df),
        "avg_price": round(df['Price'].mean(), 2),
        "avg_protein": round(df['Protein'].mean(), 2),
        "chart_data": { "names": top_protein['Food'].tolist(), "protein": top_protein['Protein'].tolist() }
    })

# --- 4. PDF GENERATOR (FIXED) ---
@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    data = request.json
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="OptiNutri Diet Plan", ln=1, align='C')
        
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        
        # FIX 1: Replace '₹' with 'Rs.' (Prevents Crash)
        total_cost = str(data.get('total_cost', 'N/A')).replace('₹', 'Rs. ')
        pdf.cell(200, 10, txt=f"Total Daily Cost: {total_cost}", ln=1, align='L')
        pdf.ln(5)
        
        # Table Header
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(100, 10, txt="Food Item", border=1, fill=True)
        pdf.cell(40, 10, txt="Quantity", border=1, fill=True)
        pdf.cell(40, 10, txt="Cost", border=1, fill=True, ln=1)
        
        # Table Rows
        pdf.set_font("Arial", size=11)
        for item in data.get('plan', []):
            # FIX 2: Sanitize inputs to prevent encoding errors
            food = str(item['food']).encode('latin-1', 'replace').decode('latin-1')
            qty = str(item['qty'])
            cost = str(item['cost']).replace('₹', 'Rs. ')
            
            pdf.cell(100, 10, txt=food, border=1)
            pdf.cell(40, 10, txt=qty, border=1)
            pdf.cell(40, 10, txt=cost, border=1, ln=1)
            
        # Save to temp directory to avoid permission issues
        pdf_file = os.path.join(tempfile.gettempdir(), "diet_plan.pdf")
        pdf.output(pdf_file)
        
        return send_file(pdf_file, as_attachment=True, download_name="OptiNutri_Plan.pdf")
        
    except Exception as e:
        print(f"PDF GENERATION ERROR: {e}") # Check your terminal if it fails again
        return jsonify({"error": "PDF Failed"}), 500

# --- 5. OTHER ROUTES (BMI, VOICE, VISION) ---
@app.route('/calculate-bmi', methods=['POST'])
def calculate_bmi():
    data = request.json
    try:
        w = float(data.get('weight'))
        h = float(data.get('height'))
        a = int(data.get('age'))
        g = data.get('gender')
        if g == 'male': bmr = (10 * w) + (6.25 * h) - (5 * a) + 5
        else: bmr = (10 * w) + (6.25 * h) - (5 * a) - 161
        return jsonify({"bmi": round(w/((h/100)**2), 1), "status": "Normal", "calories": round(bmr*1.375), "protein": round(w*1.5)})
    except: return jsonify({"error": "Invalid"}), 400

# --- REPLACE /voice-chat AND ADD /generate-audio ---

# --- REPLACEMENT BLOCK FOR SERVER.PY ---

# --- REPLACE THIS FUNCTION IN SERVER.PY ---
@app.route('/voice-chat', methods=['POST'])
def voice_chat():
    data = request.json
    try:
        api_key = os.getenv("GROQ_API_KEY")
        client = ChatGroq(api_key=api_key, model_name="llama-3.3-70b-versatile")
        
        user_text = data.get('text')
        lang = data.get('language')
        
        # IMPROVED PROMPT: Asks for bullet points and clear structure
        system_prompt = f"""
        You are an expert Nutritionist. The user asked: "{user_text}".
        Answer in {lang}. 
        Rules:
        1. Give a detailed answer with bullet points for steps/ingredients.
        2. Do NOT use asterisks (*) or markdown symbols like #.
        3. Keep it strictly text-based and easy to read.
        """
        
        res = client.invoke(system_prompt).content
        
        # CLEANUP: Remove any remaining markdown symbols
        clean_res = res.replace('*', '').replace('#', '').strip()
        
        return jsonify({"reply": clean_res}) 
        
    except Exception as e:
        print(f"Voice Error: {e}")
        return jsonify({"reply": "Sorry, I encountered an error."}), 500

@app.route('/generate-audio', methods=['POST'])
def generate_audio():
    data = request.json
    try:
        text = data.get('text', '')
        # Generate Audio on Demand (Slower, but optional)
        tts = gTTS(text=text, lang='en', tld='co.in', slow=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            with open(fp.name, "rb") as f: 
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
                
        return jsonify({"audio": audio_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files: return jsonify({"error": "No image"}), 400
        file = request.files['image']
        b64 = base64.b64encode(file.read()).decode('utf-8')
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        comp = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Identify ingredients."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
            temperature=0.7, max_tokens=600
        )
        return jsonify({"analysis": comp.choices[0].message.content})
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)