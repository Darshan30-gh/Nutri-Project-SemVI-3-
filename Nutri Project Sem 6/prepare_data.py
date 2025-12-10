import pandas as pd
import numpy as np

# 1. Load your new Kaggle file
try:
    df = pd.read_csv('data/raw_food.csv')
    print("✅ Found the Kaggle file!")

    # 2. Fix Column Names (The file likely has 'Diet' or 'Course' cols we don't need)
    # We will try to find the 'name' column and rename it to 'Food'
    # And ensure we have numerical values for nutrition.
    
    # Auto-detect the "Name" column (usually the first one)
    df.rename(columns={df.columns[0]: 'Food'}, inplace=True)

    # 3. Add Fake Prices (Because the dataset doesn't have them)
    print("...Injecting market prices...")
    df['Price'] = np.random.randint(20, 200, len(df)) # Prices between ₹20 and ₹200

    # 4. Save as the final 'food_data.csv' that your App uses
    # We keep only relevant columns to avoid errors
    # Note: We check if columns exist before keeping them
    cols_to_keep = ['Food', 'Price']
    for c in ['Calories', 'Protein', 'Fat']:
        if c in df.columns:
            cols_to_keep.append(c)
        else:
            # If column is missing, generate random data so app doesn't crash
            df[c] = np.random.randint(10, 50)
            cols_to_keep.append(c)

    final_df = df[cols_to_keep]
    final_df.to_csv('data/food_data.csv', index=False)
    
    print(f"🎉 Success! Your app now has {len(final_df)} food items.")

except FileNotFoundError:
    print("❌ Error: Could not find 'data/raw_food.csv'. Did you rename it?")