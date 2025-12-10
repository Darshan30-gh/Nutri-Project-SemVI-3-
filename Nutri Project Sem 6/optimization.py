import pandas as pd
from scipy.optimize import linprog
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# 1. Load Environment Variables (Your API Key)
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# 2. Setup the AI Model (using the free model on Groq)
# Updated Model Name (Llama 3.3 is the newest supported version)
llm = ChatGroq(temperature=0.5, groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

# --- PART A: THE MATH (Data Science Syllabus) ---
print("--- Step 1: Running Mathematical Optimization... ---")
df = pd.read_csv('data/food_data.csv')

costs = df['Price'].values
A_ub = [-df['Calories'].values, -df['Protein'].values, -df['Fat'].values]
b_ub = [-2000, -50, -20] # Constraints: 2000 cal, 50g protein, 20g fat
bounds = [(0, 5) for _ in range(len(df))]

result = linprog(c=costs, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

if result.success:
    # Prepare the list of ingredients for the AI
    ingredients_list = []
    total_cost = 0
    print("\n--- OPTIMIZED INGREDIENTS (Calculated by Linear Algebra) ---")
    for i, qty in enumerate(result.x):
        if qty > 0:
            item_name = df.iloc[i]['Food']
            grams = qty * 100
            ingredients_list.append(f"{item_name}: {grams:.0f}g")
            print(f" - {item_name}: {grams:.0f}g")
            total_cost += qty * df.iloc[i]['Price']
    
    print(f"Total Daily Cost: ₹{total_cost:.2f}")

    # --- PART B: THE AI AGENT (GenAI Integration) ---
    print("\n--- Step 2: AI Chef is thinking... ---")
    
    # This is the "Prompt" we send to the AI
    prompt = f"""
    You are a creative chef for students on a budget.
    I have mathematically calculated the cheapest ingredients to survive today:
    {', '.join(ingredients_list)}

    Please write a tasty Indian-style recipe using ONLY these ingredients. 
    Give it a fun name. Keep it simple.
    """
    
    # Get the response from AI
    response = llm.invoke(prompt)
    
    print("\n" + "="*40)
    print(response.content) # This prints the AI's recipe
    print("="*40)

else:
    print("Optimization Failed! Check constraints.")